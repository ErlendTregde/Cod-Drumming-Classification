"""Agile modeling / vector search — Perch's intended human-in-the-loop workflow.

`src/inference/` runs a *trained classifier* and commits to a hard label per event.
This package does the complementary thing Perch is designed for: **search by
example**. Give one example (a class prototype, or a single clip), rank a long
file's detected events by embedding similarity, and let a human verify the top
results — the "agile modeling" loop (perch-hoplite). It needs no trained head.
"""
