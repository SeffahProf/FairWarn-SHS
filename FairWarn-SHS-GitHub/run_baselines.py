import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

commands = [
    [sys.executable, str(ROOT / "src" / "01_inspect_data.py")],
    [sys.executable, str(ROOT / "src" / "02_train_baselines.py")],
]

for command in commands:
    print("\n" + "=" * 70)
    print("RUNNING:", " ".join(command))
    print("=" * 70)
    subprocess.run(command, check=True)
