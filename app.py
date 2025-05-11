import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import community as community_louvain

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} genres")
    return df

def create_genre_similarity_network(df, similarity_threshold=0.85, top_n=50):

    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

    top_genres_df = df.sort_values('popularity', ascending=False).head(top_n)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(top_genres_df[features])

    similarity_matrix = cosine_similarity(scaled_features)

    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=top_genres_df['genres'], 
        columns=top_genres_df['genres']
    )

    G = nx.Graph()
    
    genre_list = top_genres_df['genres'].tolist()
    for genre in genre_list:
        popularity = float(top_genres_df[top_genres_df['genres'] == genre]['popularity'].values[0])
        G.add_node(genre, popularity=popularity)

    for i, genre1 in enumerate(genre_list):
        for genre2 in genre_list[i+1:]:
            similarity = similarity_df.loc[genre1, genre2]
            if similarity > similarity_threshold:
                G.add_edge(genre1, genre2, weight=similarity)
    
    print(f"Created network with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G, top_genres_df

def visualize_genre_network(G, top_genres_df, output_file='genre_similarity_network.png'):

    plt.figure(figsize=(16, 12))
    
    communities = community_louvain.best_partition(G)

    num_communities = len(set(communities.values()))
    color_map = cm.get_cmap('viridis', num_communities)
    node_colors = [color_map(communities[node]) for node in G.nodes()]

    popularity_multiplier = 1
    node_sizes = [G.nodes[node]['popularity'] * popularity_multiplier for node in G.nodes()]

    pos = nx.spring_layout(G, k=0.3, seed=42)

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)

    popularity_threshold = top_genres_df['popularity'].quantile(0.7)
    labels = {node: node for node in G.nodes() if G.nodes[node]['popularity'] >= popularity_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    plt.title(f'Genre Similarity Network (Threshold={similarity_threshold})', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')
    plt.show()
    
    print(f"Network visualization saved to {output_file}")
    
    community_summary = {}
    for node, community_id in communities.items():
        if community_id not in community_summary:
            community_summary[community_id] = []
        community_summary[community_id].append(node)
    
    print(f"Detected {num_communities} communities:")
    for community_id, genres in community_summary.items():
        genres_str = ', '.join(sorted(genres)[:5])
        if len(genres) > 5:
            genres_str += f", ... ({len(genres) - 5} more)"
        print(f"Community {community_id}: {genres_str}")
    
    return communities

def analyze_network_centrality(G):

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    centrality_df = pd.DataFrame({
        'Genre': list(G.nodes()),
        'Degree': [degree_centrality[node] for node in G.nodes()],
        'Betweenness': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector': [eigenvector_centrality[node] for node in G.nodes()],
        'Popularity': [G.nodes[node]['popularity'] for node in G.nodes()]
    })
    
    centrality_df = centrality_df.sort_values('Eigenvector', ascending=False)
    
    centrality_df.to_csv('genre_centrality_metrics.csv', index=False)
    print("Centrality metrics saved to genre_centrality_metrics.csv")
    
    return centrality_df

if __name__ == "__main__":
    file_path = 'data/data_by_genres.csv'
    similarity_threshold = 0.85
    top_n = 300
    
    df_genres = load_data(file_path)

    G, top_genres_df = create_genre_similarity_network(
        df_genres, 
        similarity_threshold=similarity_threshold,
        top_n=top_n
    )
    
    communities = visualize_genre_network(G, top_genres_df)
    
    centrality_df = analyze_network_centrality(G)
    
    print("\nTop 20 most central genres (by eigenvector centrality):")
    print(centrality_df[['Genre', 'Eigenvector', 'Popularity']].head(20).to_string(index=False))