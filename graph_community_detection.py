"""
graph_community_detection.py
--------------------------------------------------------------------------------
Task 6 (Bonus): Graph-Based Coordinated Account & Ring Detection (Amrutha's Lane)

Algorithm & Pipeline:
1. Construct User-Item Bipartite Ratings Graph and temporal co-rating network.
2. Link user pairs (u1, u2) who co-rated the same products on the same date.
3. Apply Community Detection (Connected Components & Greedy Modularity) to detect collusion rings.
4. Calculate collusion score s_collusion per community based on cluster size, co-rating intensity, and burst density.
5. Export community_scores.csv with columns: user_id, community_id, community_size, collusion_score.
--------------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

RATINGS_FILE = "recommender/data/ratings_sample.csv"
OUTPUT_FILE = "community_scores.csv"

def run_graph_detection():
    path_to_use = RATINGS_FILE if os.path.exists(RATINGS_FILE) else "attack_data/ratings_sample.csv"
    if not os.path.exists(path_to_use):
        for root, _, files in os.walk("."):
            if "ratings_sample.csv" in files:
                path_to_use = os.path.join(root, "ratings_sample.csv")
                break

    print(f"[Task 6] Loading ratings for graph construction from '{path_to_use}'...", flush=True)
    df = pd.read_csv(path_to_use)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date

    print("[1/4] Constructing temporal co-rating user graph...", flush=True)
    
    # Group users who rated the same item on the same date
    grouped = df.groupby(['item_id', 'date'])['user_id'].apply(list)
    edge_weights = {}

    for users in grouped:
        unique_users = sorted(set(users))
        if len(unique_users) > 1 and len(unique_users) <= 50:
            for u1, u2 in combinations(unique_users, 2):
                edge_weights[(u1, u2)] = edge_weights.get((u1, u2), 0) + 1

    print(f"      Extracted {len(edge_weights):,} temporal co-rating user pairs.", flush=True)

    # Build NetworkX Graph
    G = nx.Graph()
    all_users = df['user_id'].unique()
    G.add_nodes_from(all_users)

    for (u1, u2), w in edge_weights.items():
        G.add_edge(u1, u2, weight=w)

    print("[2/4] Running Community Detection on co-rating graph...", flush=True)
    
    # Find connected components of co-rating network with edge weight >= 1
    # Filter subgraphs with edges for active co-rating communities
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1]
    
    community_records = []
    comm_counter = 1

    # Map isolated users to default community 0
    isolated_users = set(G.nodes()) - set().union(*[s.nodes() for s in subgraphs])
    for u in isolated_users:
        community_records.append({
            "user_id": u,
            "community_id": 0,
            "community_size": 1,
            "collusion_score": 0.0000
        })

    print(f"[3/4] Scoring {len(subgraphs):,} detected co-rating communities...", flush=True)

    for sg in subgraphs:
        c_size = sg.number_of_nodes()
        total_edges = sg.number_of_edges()
        max_possible_edges = (c_size * (c_size - 1)) / 2 if c_size > 1 else 1
        density = total_edges / max_possible_edges
        avg_weight = np.mean([d['weight'] for _, _, d in sg.edges(data=True)]) if total_edges > 0 else 0

        # Calculate collusion score based on community size, density, and co-rating strength
        # Higher score = suspicious coordinated account ring
        raw_score = (avg_weight * 0.40) + (density * 0.40) + (min(c_size, 20) / 20.0 * 0.20)
        collusion_score = float(np.clip(raw_score, 0.0, 1.0))

        for u in sg.nodes():
            community_records.append({
                "user_id": u,
                "community_id": comm_counter,
                "community_size": c_size,
                "collusion_score": round(collusion_score, 4)
            })
        comm_counter += 1

    comm_df = pd.DataFrame(community_records)

    print("\n" + "=" * 75)
    print("TASK 6: GRAPH COMMUNITY DETECTION SUMMARY")
    print("=" * 75)
    print(comm_df[comm_df['community_id'] > 0].sort_values(by='collusion_score', ascending=False).head(10).to_string(index=False))
    print("-" * 75)
    print(f"Total Users Analyzed: {len(comm_df):,}")
    print(f"Active Co-Rating Communities Detected: {comm_counter - 1:,}")
    print(f"Users in Coordinated Rating Rings: {(comm_df['community_id'] > 0).sum():,}")
    print("=" * 75 + "\n")

    comm_df.to_csv(OUTPUT_FILE, index=False)
    print(f"[4/4] Saved community scores to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_graph_detection()
