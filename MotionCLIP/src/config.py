import os
from pathlib import Path

# Resolve paths relative to this file's location (MotionCLIP/src/config.py)
# REPO_ROOT/src/config.py -> REPO_ROOT is parents[1]
# Models are expected at MotionCLIP/models/... 
# But wait, looking at repo structure, MotionCLIP is the repo root.
# So if config is at MotionCLIP/src/config.py, then models is at MotionCLIP/models
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent # MotionCLIP root

SMPL_DATA_PATH = str(PROJECT_ROOT / "models/smpl")
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')

SMPLH_AMASS_PATH = str(PROJECT_ROOT / 'models/smplh')
SMPLH_AMASS_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "neutral/model.npz")
SMPLH_AMASS_MALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "male/model.npz")
SMPLH_AMASS_FEMALE_MODEL_PATH = os.path.join(SMPLH_AMASS_PATH, "female/model.npz")

SMPLX_DATA_PATH = str(PROJECT_ROOT / "models/smplx")
SMPLX_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_NEUTRAL.pkl")
SMPLX_MALE_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_MALE.pkl")
SMPLX_FEMALE_MODEL_PATH = os.path.join(SMPLX_DATA_PATH, "SMPLX_FEMALE.pkl")

ROT_CONVENTION_TO_ROT_NUMBER = {
    'legacy': 23,
    'no_hands': 21,
    'full_hands': 51,
    'mitten_hands': 33,
}

GENDERS = ['neutral', 'male', 'female']
NUM_BETAS = 10
