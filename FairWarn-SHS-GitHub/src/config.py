from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / "data" / "processed" / "FairWarn_SHS_Node_Features.csv"
EDGE_FILE = ROOT / "data" / "processed" / "FairWarn_SHS_Edge_List.csv"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"
OUTPUT_MODELS = ROOT / "outputs" / "models"
OUTPUT_LOGS = ROOT / "outputs" / "logs"

SEEDS = [42, 123, 456, 789, 1010]
N_SPLITS = 5
N_REPEATS = 5
