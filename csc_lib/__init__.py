"""Stage-1 CSC library — labeling, model, features, evaluation.

This package lives at the project root under ``lib/`` so that
runnable scripts in ``tools/`` and notebooks can import it as
``lib.csc.labeling`` etc. The installed package
``csc_uav_tracking`` (under ``src/``) holds dataset loaders,
telemetry schema, and registry; ``lib/`` holds the Stage-1
pipeline that consumes those.
"""
