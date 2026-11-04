"""Stable selection seeds and shared budget validation."""
import hashlib

def seed_for(pair_id, seed=20261031):
    return int.from_bytes(hashlib.sha256(f'{seed}:{pair_id}'.encode()).digest()[:8], 'big') % (2**63-1)

def budget_matches(before, after, ratio, tolerance=.005):
    return before > 0 and abs(after / before - ratio) <= tolerance
