"""eval.py — SALT-RD model evaluation.

Will implement per-head and per-dataset evaluation of a trained SALTRD
checkpoint against the val/diagnostic NPZ splits.  Reports AUC-ROC,
precision/recall, and calibration curves per label.
"""

# TODO: implement after train.py produces a valid checkpoint and the val/
#       diagnostic splits have been verified for label balance.
