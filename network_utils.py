import networkx as nx
from itertools import combinations
from typing import Dict, Set
import pandas as pd
import random
from networkx import edge_boundary
import community as community_louvain  # Louvain method



def create_co_mention_graph(names_df: pd.DataFrame, min_weight: int = 2):
    """
    Creates a co-mention graph from person entities in articles.
    
    Args:
        names_df: DataFrame containing 'canonical_name' and 'article_ids' columns
        min_weight: Minimum number of shared articles for an edge to be included
        
    Returns:
        A NetworkX Graph where:
        - Nodes represent unique people
        - Edges represent co-mentions in articles
        - Edge weights indicate number of shared articles
    """
    
    # Step 1: Create the full graph
    G = nx.Graph()

    for _, row in names_df.iterrows():
        G.add_node(row["canonical_name"])

    for (_, row1), (_, row2) in combinations(names_df.iterrows(), 2):
        shared_articles = set(row1["article_ids"]) & set(row2["article_ids"])
        weight = len(shared_articles)
        if weight > 0:
            G.add_edge(row1["canonical_name"], row2["canonical_name"], weight=weight)

    # Step 2: Keep only edges with weight > 2
    edges_to_keep = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d["weight"] > 2
    ]

    # Create filtered graph H from those edges
    H = nx.Graph()
    H.add_edges_from([(u, v, d) for u, v, d in edges_to_keep])

    # Step 3: Remove nodes with only one neighbor and any resulting isolates
    nodes_degree_1 = [n for n in H.nodes if H.degree(n) == 1]
    H.remove_nodes_from(nodes_degree_1)
    H.remove_nodes_from(list(nx.isolates(H)))  # Removes any new isolates formed
    
    return H

def network_summary(graph: nx.Graph):
    """Generate detailed network statistics with visual formatting"""
    
    # Basic stats
    stats = {
        "Nodes": len(graph.nodes()),
        "Edges": len(graph.edges()),
        "Average Degree": sum(dict(graph.degree()).values()) / len(graph.nodes())
    }
    
    # Degree distribution
    degrees = dict(graph.degree())
    degree_df = pd.DataFrame.from_dict(degrees, orient='index', columns=['Degree'])
    
    # Centrality measures
    centrality = {
        "Degree Centrality": nx.degree_centrality(graph),
        "Betweenness Centrality": nx.betweenness_centrality(graph),
        "Closeness Centrality": nx.closeness_centrality(graph)
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "Metric": list(stats.keys()),
        "Value": list(stats.values())
    })
    
    # Formatting
    summary_df['Value'] = summary_df['Value'].apply(
        lambda x: f"{x:.3f}" if isinstance(x, float) else x
    )
    
    return summary_df, degree_df, centrality



import numpy as np
def louvain_based_communities_randomized(
    G,
    runs=10,
    top_k=10,
    weight_threshold=10,
    cohesion_ratio=0.5,
    seed=None,
    res=1.5
):
    """Run Louvain detection multiple times with deterministic behavior."""

    rng = np.random.default_rng(seed)

    best_partition = {}
    best_score = -1

    edges_all = list(G.edges(data=True))
    edges_all.sort()  # ensure deterministic edge order before shuffle

    for run_idx in range(runs):
        edges = edges_all.copy()
        rng.shuffle(edges)

        filtered_edges = [(u, v) for u, v, d in edges if d.get('weight', 0) >= weight_threshold]
        G_filtered = G.edge_subgraph(filtered_edges).copy()

        if len(G_filtered) == 0:
            continue

        partition = community_louvain.best_partition(
            G_filtered,
            weight='weight',
            random_state=rng.integers(0, 1e6),
            resolution=res
        )

        community_map = {}
        for node, comm_id in partition.items():
            community_map.setdefault(comm_id, set()).add(node)
        communities = list(community_map.values())

        score = community_louvain.modularity(partition, G_filtered, weight='weight')
        if score > best_score:
            best_score = score
            best_partition = communities

    processed = []
    assigned = set()

    for comm in best_partition:
        members = set(comm) - assigned
        if not members:
            continue

        seed_node = max(members, key=lambda n: G.degree(n, weight='weight'))

        valid_members = {
            n for n in members if (
                (internal := sum(G[n][nbr].get('weight', 0) for nbr in members & set(G[n]))) /
                (internal + sum(G[n][nbr].get('weight', 0) for nbr in set(G[n]) - members) + 1e-9)
            ) >= cohesion_ratio
        }

        final_community = {
            n for n in valid_members if len(set(G.neighbors(n)) & valid_members) >= 2
        }

        if final_community:
            processed.append({
                'seed': seed_node,
                'members': final_community,
                'size': len(final_community),
                'internal_edges': sum(G[u][v]['weight'] for u, v in G.subgraph(final_community).edges())
            })
            assigned.update(final_community)

        if len(processed) >= top_k:
            break

    return sorted(processed, key=lambda x: (-x['internal_edges'] / x['size'], -x['size']))
