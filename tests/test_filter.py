"""Tests that protect the filter's correctness. Fill in as you implement filter.py.

The two invariants worth writing first:
  1. Synthetic recovery: generate matches from KNOWN serve/return skills, run the
     filter, assert it converges back to those skills.
  2. Zero-drift sanity: with gamma=0 the filter should match a static logistic
     regression fit on the same data.
"""
