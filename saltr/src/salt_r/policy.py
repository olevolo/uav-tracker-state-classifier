"""policy.py — SALT-RD runtime decision policy.

Will implement the lightweight inference wrapper that converts SALTRD output
probabilities into concrete tracker control decisions:
  - full SGLATrack inference vs. fast path
  - search-region expansion triggers
  - re-initialisation recommendations
  - dynamic scene class override

Design constraint: must run in < 0.5 ms on CPU (MBP M2) after GRU forward pass.
"""

# TODO: implement after:
#   1. eval.py confirms AUC > 0.75 on val split for needs_full_compute head
#   2. Operating-point thresholds are calibrated on val split
#   3. Integration contract with integrate.py is defined
