"""train.py — SALT-RD training loop.

Will implement supervised training of the SALTRD GRU model on the NPZ dataset
produced by collect_features.py.  Supports multi-label BCE loss with per-head
weighting, sequence-length batching, and W&B/TensorBoard logging.
"""

# TODO: implement after:
#   1. collect_features.py collection loop is complete and NPZ is verified
#   2. SALTRD model.py architecture is finalised
#   3. Feature normalisation statistics (mean/std) are computed from training split
