import sys
from pathlib import Path

# Add demo/ to sys.path so `from eval.common import ...` works,
# mirroring the established pattern in demo/generate_dataset.py:20.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
