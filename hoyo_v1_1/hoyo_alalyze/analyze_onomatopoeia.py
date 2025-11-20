import json
import os
from collections import Counter, defaultdict


def load_all_annotations(root: str):
    """
    HOYO の *.json を全部読み込んで annotation を集計。
    """
    files = [f for f in os.listdir(root) if f.endswith(".json")]
    files.sort(key=lambda x: int(x.replace(".json", "")))

    instruction_counts = Counter()
    freestyle_counts = Counter()
    select_based_counts = Counter()

    for fname in files:
        path = os.path.join(root, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ann = data.get("annotation", {})
        instr = ann.get("instruction")
        if instr:
            instruction_counts[instr] += 1

        freestyle = ann.get("freestyle", []) or []
        for w in freestyle:
            freestyle_counts[w] += 1

        # select は 10人の被験者が選んだインデックス（freestyle の index）とみなす
        select = ann.get("select", []) or []
        for idx in select:
            if 0 <= idx < len(freestyle):
                w = freestyle[idx]
                select_based_counts[w] += 1

    return instruction_counts, freestyle_counts, select_based_counts


def main():
    root = os.path.dirname(__file__)
    instruction_counts, freestyle_counts, select_based_counts = load_all_annotations(root)

    print("=== HOYO オノマトペ集計 ===")
    print(f"  instruction 種類数: {len(instruction_counts)}")
    print(f"  freestyle 種類数   : {len(freestyle_counts)}")
    print(f"  select ベース種類数: {len(select_based_counts)}")
    print()

    def show_top(counter, name, topn=30):
        if topn is None:
            print(f"--- All {name} ({len(counter)} 種類) ---")
            items = counter.most_common()
        else:
            print(f"--- Top {topn} {name} ---")
            items = counter.most_common(topn)
        for w, c in items:
            print(f"{w}\t{c}")
        print()

    # 代表だけざっくり
    show_top(instruction_counts, "instruction (top)", topn=30)
    show_top(freestyle_counts, "freestyle（候補として出現回数, top）", topn=30)
    show_top(select_based_counts, "select ベース（被験者に選ばれた回数, top）", topn=30)

    # フルリストはファイルに保存
    out_dir = root
    os.makedirs(out_dir, exist_ok=True)
    instr_path = os.path.join(out_dir, "onomatopoeia_instruction_all.txt")
    select_path = os.path.join(out_dir, "onomatopoeia_select_all.txt")
    freestyle_path = os.path.join(out_dir, "onomatopoeia_freestyle_all.txt")

    with open(instr_path, "w", encoding="utf-8") as f:
        for w, c in instruction_counts.most_common():
            f.write(f"{w}\t{c}\n")
    with open(select_path, "w", encoding="utf-8") as f:
        # select は固定 10カテゴリの得票数なので、ここでは
        # readme.txt に書かれている順で出力する
        base_select_labels = [
            "すたすた",
            "せかせか",
            "てくてく",
            "どっしどっし",
            "とぼとぼ",
            "のしのし",
            "のろのろ",
            "ぶらぶら",
            "よたよた",
            "よろよろ",
        ]
        for w in base_select_labels:
            c = select_based_counts.get(w, 0)
            f.write(f"{w}\t{c}\n")
    with open(freestyle_path, "w", encoding="utf-8") as f:
        for w, c in freestyle_counts.most_common():
            f.write(f"{w}\t{c}\n")

    print("Saved full instruction list to:", instr_path)
    print("Saved full fixed-10 select label list to:", select_path)
    print("Saved full freestyle list to:", freestyle_path)


if __name__ == "__main__":
    main()


