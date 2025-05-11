import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import community as community_louvain

# Load the genres data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} genres")
    return df

# Function to create genre similarity network
def create_genre_similarity_network(df, similarity_threshold=0.85, top_n=50):
    """
    Creates a network where:
    - Nodes are music genres
    - Edges connect genres with similarity above threshold
    - Node size is based on genre popularity
    - Colors represent communities detected
    
    Parameters:
    - df: DataFrame containing genre data
    - similarity_threshold: Minimum similarity to create an edge
    - top_n: Number of top genres by popularity to include
    """
    # Select relevant audio features for similarity calculation
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    # Select top N genres by popularity
    top_genres_df = df.sort_values('popularity', ascending=False).head(top_n)
    
    # Standardize the features to have mean=0 and variance=1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(top_genres_df[features])
    
    # Calculate cosine similarity between genres
    similarity_matrix = cosine_similarity(scaled_features)
    
    # Create a similarity DataFrame for easier indexing
    similarity_df = pd.DataFrame(
        similarity_matrix, 
        index=top_genres_df['genres'], 
        columns=top_genres_df['genres']
    )
    
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes with popularity as an attribute
    genre_list = top_genres_df['genres'].tolist()
    for genre in genre_list:
        popularity = float(top_genres_df[top_genres_df['genres'] == genre]['popularity'].values[0])
        G.add_node(genre, popularity=popularity)
    
    # Add edges based on similarity threshold
    for i, genre1 in enumerate(genre_list):
        for genre2 in genre_list[i+1:]:
            similarity = similarity_df.loc[genre1, genre2]
            if similarity > similarity_threshold:
                G.add_edge(genre1, genre2, weight=similarity)
    
    print(f"Created network with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G, top_genres_df

# Function to visualize the network
def visualize_genre_network(G, top_genres_df, output_file='genre_similarity_network.png'):
    """
    Visualizes the genre similarity network with:
    - Node size based on popularity
    - Colors based on community detection
    - Labels for most popular genres
    """
    plt.figure(figsize=(16, 12))
    
    # Detect communities
    communities = community_louvain.best_partition(G)
    
    # Convert communities to colors
    num_communities = len(set(communities.values()))
    color_map = cm.get_cmap('viridis', num_communities)
    node_colors = [color_map(communities[node]) for node in G.nodes()]
    
    # Scale node sizes based on popularity (adjust multiplier as needed)
    popularity_multiplier = 1  # Reduced from 40 to make nodes smaller
    node_sizes = [G.nodes[node]['popularity'] * popularity_multiplier for node in G.nodes()]
    
    # Create layout
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    # Draw network elements
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
    
    # Only label nodes with higher popularity for clarity
    popularity_threshold = top_genres_df['popularity'].quantile(0.7)
    labels = {node: node for node in G.nodes() if G.nodes[node]['popularity'] >= popularity_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Add a title
    plt.title(f'Genre Similarity Network (Threshold={similarity_threshold})', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=1200, bbox_inches='tight')
    plt.show()
    
    print(f"Network visualization saved to {output_file}")
    
    # Create a summary of communities
    community_summary = {}
    for node, community_id in communities.items():
        if community_id not in community_summary:
            community_summary[community_id] = []
        community_summary[community_id].append(node)
    
    # Print community information
    print(f"Detected {num_communities} communities:")
    for community_id, genres in community_summary.items():
        genres_str = ', '.join(sorted(genres)[:5])  # Show first 5 genres in each community
        if len(genres) > 5:
            genres_str += f", ... ({len(genres) - 5} more)"
        print(f"Community {community_id}: {genres_str}")
    
    return communities

# Function to analyze centrality metrics
def analyze_network_centrality(G):
    """
    Calculates various centrality metrics for the network and
    returns a table of the most central genres
    """
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Create a DataFrame
    centrality_df = pd.DataFrame({
        'Genre': list(G.nodes()),
        'Degree': [degree_centrality[node] for node in G.nodes()],
        'Betweenness': [betweenness_centrality[node] for node in G.nodes()],
        'Eigenvector': [eigenvector_centrality[node] for node in G.nodes()],
        'Popularity': [G.nodes[node]['popularity'] for node in G.nodes()]
    })
    
    # Sort by eigenvector centrality (most influential nodes)
    centrality_df = centrality_df.sort_values('Eigenvector', ascending=False)
    
    # Save to CSV
    centrality_df.to_csv('genre_centrality_metrics.csv', index=False)
    print("Centrality metrics saved to genre_centrality_metrics.csv")
    
    return centrality_df

# Main execution
if __name__ == "__main__":
    # Parameters
    file_path = 'data/data_by_genres.csv'
    similarity_threshold = 0.85  # Adjust based on desired network density
    top_n = 300  # Number of top genres to include
    
    # Load data
    df_genres = load_data(file_path)
    
    # Create network
    G, top_genres_df = create_genre_similarity_network(
        df_genres, 
        similarity_threshold=similarity_threshold,
        top_n=top_n
    )
    
    # Visualize network
    communities = visualize_genre_network(G, top_genres_df)
    
    # Analyze centrality
    centrality_df = analyze_network_centrality(G)
    
    # Print top 20 genres by centrality
    print("\nTop 20 most central genres (by eigenvector centrality):")
    print(centrality_df[['Genre', 'Eigenvector', 'Popularity']].head(20).to_string(index=False))