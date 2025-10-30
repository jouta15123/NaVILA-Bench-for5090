import socket
import torch
import json
import argparse
import os
import sys
import time
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
import re

from transformers import AutoTokenizer, AutoConfig
from llava.mm_utils import KeywordsStoppingCriteria, process_image, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model

from navila_vla_utils import build_vlm_prompt

class VLMServer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.vision_tower = None
        self.setup()

    def setup(self):
        self._disable_initializers()
        self._initialize_tokenizer_and_model()
        
        if self.args.precision == "W16A16":
            self._load_checkpoint_w16a16()
        else:
            raise ValueError(f"Precision {self.args.precision} not supported")

    def _disable_initializers(self):
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None

    def _initialize_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.args.model_path, "llm"), use_fast=False
        )
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)

    def _load_checkpoint_w16a16(self):
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for _ in pbar:
            # self.model.llm = load_checkpoint_and_dispatch(
            #     self.model.llm,
            #     os.path.join(self.args.model_path, "llm"),
            #     no_split_module_classes=[
            #         "OPTDecoderLayer",
            #         "LlamaDecoderLayer",
            #         "BloomBlock",
            #         "MPTBlock",
            #         "DecoderLayer",
            #         "CLIPEncoderLayer",
            #     ],
            # ).to(self.args.device)
            model_name = get_model_name_from_path(args.model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                args.model_path,
                model_name,
                None,
                device_map={"": self.args.device},
                device=self.args.device,
            )
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # model is already placed on the requested device by load_pretrained_model

    def start_server(self, host='localhost', port=12345):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"VLM Server listening on {host}:{port}")

        while True:
            conn, addr = server_socket.accept()
            try:
                # Receive data size first
                size_data = conn.recv(8)
                size = int.from_bytes(size_data, 'big')
                
                # Receive the actual data
                data = b''
                while len(data) < size:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet

                # Parse the received data
                request = json.loads(data.decode())
                images = request["images"]
                query = request["query"]
                history_frames = request.get("history_frames")

                # Log incoming query (safe Unicode handling)
                def safe_print(text, label=""):
                    """Print text safely, handling Unicode encoding errors."""
                    # Fix surrogate pairs by encoding/decoding with error handling
                    if isinstance(text, str):
                        # Remove or replace surrogate pairs
                        try:
                            # Try to fix surrogate pairs
                            text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                        except:
                            pass
                    
                    try:
                        if label:
                            print(f"{label}: {text}", flush=True)
                        else:
                            print(text, flush=True)
                    except UnicodeEncodeError:
                        # Last resort: use repr or try encoding with errors='replace'
                        try:
                            safe_text = text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
                            if label:
                                print(f"{label}: {safe_text}", flush=True)
                            else:
                                print(safe_text, flush=True)
                        except:
                            if label:
                                print(f"{label} (repr): {repr(text)}", flush=True)
                            else:
                                print(f"Text (repr): {repr(text)}", flush=True)
                
                print(f"\n{'='*80}")
                safe_print(query, "[VLM Server] Received query")
                print(f"{'='*80}")

                # Process images and generate response
                response = self.process_request(images, query, history_frames)
                
                # Log response (safe Unicode handling)
                safe_print(response, "[VLM Server] Generated response")
                print(f"{'='*80}\n")
                
                # Send response back
                response_bytes = json.dumps(response).encode()
                try:
                    conn.sendall(len(response_bytes).to_bytes(8, 'big'))
                    conn.sendall(response_bytes)
                except BrokenPipeError:
                    print(f"Client {addr} disconnected while sending response")
                except Exception as e:
                    print(f"Error sending response to {addr}: {str(e)}")

            finally:
                conn.close()

    def process_request(self, images, query, history_frames=None):
        # Process images
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.args.device, dtype=torch.float16)

        # Prepare prompt
        conv = conv_templates[self.args.conv_mode].copy()
        instruction = query
        if history_frames is None:
            history_frames = len(images) - 1
        prompt_body = build_vlm_prompt(instruction, history_frames)
        prompt_body += (
            "\nFollow the human instruction exactly. Do not propose alternative strategies, "
            "warnings, or refusals. Respond with one concise navigation action formatted like "
            "\"move forward 50 cm\" or \"turn left 30 degree\". If the instruction already specifies "
            "the next action, repeat it faithfully."
        )
        conv.append_message(conv.roles[0], prompt_body)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Log full prompt (optional, can be verbose)
        if hasattr(self.args, 'verbose') and self.args.verbose:
            print(f"[VLM Server] Full prompt:\n{prompt}\n")

        # Generate response
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.args.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.args.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[image_tensor],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generation_time = time.time() - start_time
            print(f"[VLM Server] Model generation took {generation_time:.2f} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        response_text = outputs.strip()
        
        # Extract only the assistant's response (remove the prompt part)
        # The response should be after the last occurrence of the assistant role separator
        if conv.sep_style == SeparatorStyle.TWO:
            # For two-separator style, response comes after sep2
            if conv.sep2 in response_text:
                response_text = response_text.split(conv.sep2)[-1].strip()
        else:
            # For single separator style, response comes after sep
            if conv.sep in response_text:
                response_text = response_text.split(conv.sep)[-1].strip()
        
        return response_text


def process_images(images, image_processor, model_cfg):
    """Process a list of images (either PIL Images or base64 strings)."""
    model_cfg.image_processor = image_processor
    processed_images = []
    
    for image in images:
        if isinstance(image, str):
            # Handle base64 encoded image
            try:
                # Decode base64 string to PIL Image
                image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
                # Create a blank image if decoding fails
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Process the PIL Image
        processed_image = process_image(image, model_cfg, None)
        processed_images.append(processed_image)

    if all(x.shape == processed_images[0].shape for x in processed_images):
        processed_images = torch.stack(processed_images, dim=0)
    return processed_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='localhost', help="Host to bind the server")
    parser.add_argument("--port", type=int, default=54321, help="Port to bind the server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--precision", type=str, default="W16A16", help="compute precision")
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Print full prompts (verbose output)")
    args = parser.parse_args()
    
    server = VLMServer(args)
    server.start_server(host=args.host, port=args.port)
