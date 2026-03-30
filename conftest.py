"""Ensure MDL root is on sys.path for tests.

When MDL/ is nested inside a parent repo (as a submodule), pytest may add
the parent to sys.path, causing the parent's src/ to shadow ours. This
conftest removes the parent path and ensures our root comes first.
"""
import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_root)

sys.path = [p for p in sys.path if os.path.abspath(p) != _parent]

if _root not in sys.path:
    sys.path.insert(0, _root)

for key in list(sys.modules.keys()):
    if key == "src" or key.startswith("src."):
        del sys.modules[key]
