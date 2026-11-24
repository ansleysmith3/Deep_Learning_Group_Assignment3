"""Shared settings for the data pipeline. Change SVHN_DIR here or set it as an environment variable."""
import os

# Folder that holds train/ and test/ (each with PNGs and digitStruct.json).
SVHN_DIR = os.environ.get("SVHN_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "SVHN"))

# Where preprocess.py saves its output and load_data.py reads it.
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

IMG_SIZE = 64        # every crop is resized to IMG_SIZE x IMG_SIZE
MARGIN = 0.30        # grow the digit box by 30% on each axis before cropping
MAX_DIGITS = 5       # k: longest number we model (see explore_data.py)
BLANK = 10           # digit-head class meaning "no digit in this position"
VAL_FRACTION = 0.10  # share of train held out for validation
SEED = 42
