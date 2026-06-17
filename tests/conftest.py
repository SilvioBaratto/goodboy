import sys
from pathlib import Path

# Add demo/ to sys.path so `import compare`, `import train_dpo` etc. work from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
