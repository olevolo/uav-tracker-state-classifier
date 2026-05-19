"""integrate.py — SALT-RD integration with frozen SGLATracker.

Will implement the thin adapter layer that:
  1. Hooks into SGLATracker telemetry extraction (read-only, no mutations to
     src/uav_tracker/) to obtain the 28 FEATURE_NAMES scalars per frame.
  2. Maintains the rolling GRU hidden state across frames.
  3. Calls policy.py to translate SALTRD predictions into tracker overrides.
  4. Exposes a drop-in SALTRDTracker wrapper compatible with the existing
     eval harness in src/uav_tracker/eval/.

IMPORTANT: src/uav_tracker/ is FROZEN — this module must never modify files
under that path.  All hooks must be non-invasive (subclassing or monkey-patching
in test/eval contexts only).
"""

# TODO: implement after:
#   1. policy.py operating points are calibrated
#   2. Telemetry extraction hooks are designed (inspect SGLATracker internals)
#   3. Integration contract is approved against FROZEN baseline benchmarks
