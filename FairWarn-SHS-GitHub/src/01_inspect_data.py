import pandas as pd
from config import NODE_FILE, EDGE_FILE

def main() -> None:
    nodes = pd.read_csv(NODE_FILE)
    edges = pd.read_csv(EDGE_FILE)

    labelled = nodes[
        nodes["Label_Available"].eq(1) &
        nodes["TARGET_AtRisk"].notna()
    ].copy()

    print("=" * 60)
    print("FAIRWARN-SHS DATA INSPECTION")
    print("=" * 60)
    print(f"All student nodes: {len(nodes)}")
    print(f"Labelled students: {len(labelled)}")
    print(f"Unlabelled students: {len(nodes) - len(labelled)}")
    print(f"Peer-study edges: {len(edges)}")
    print("\nTarget counts:")
    print(labelled["TARGET_AtRisk"].astype(int).value_counts().sort_index())
    print("\nAt-risk prevalence:")
    print(round(labelled["TARGET_AtRisk"].mean(), 4))
    print("\nMissing values by column:")
    print(nodes.isna().sum().sort_values(ascending=False).head(20))

if __name__ == "__main__":
    main()
