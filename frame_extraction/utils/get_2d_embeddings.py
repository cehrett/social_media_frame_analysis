import pandas as pd
import json
import numpy as np
from umap import UMAP

def get_2d_embeddings(embeddings_path, frames_path, clusters_path=None, umap_dim=2):
    """
    Get 2D embeddings for frames using UMAP.

    Parameters:
    embeddings_path (str): Path to the JSON file containing the embeddings.
    frames_path (str): Path to the CSV file containing the frame data.
    clusters_path (str): Path to save the CSV file with the 2D embeddings.
    umap_dim (int): Number of dimensions for UMAP embedding (default is 2).
    id_col (str): Name of the column containing the frame IDs (default is 'id').
    """
    # Load the CSV file with frame data
    df = pd.read_csv(frames_path)

    # Set the path to save the 2D embeddings
    if clusters_path is None:
        clusters_path = frames_path

    # Load the JSON file with embeddings
    with open(embeddings_path, 'r') as f:
        embeddings_dict = json.load(f)

    # Create a list of embeddings in the same order as the 'frames' column
    embeddings_list = [embeddings_dict[frame] for frame in df['frames']]

    # Convert the list of embeddings to a numpy array
    embeddings_array = np.array(embeddings_list)

    # Initialize and fit UMAP
    umap_model = UMAP(n_components=umap_dim, random_state=42)
    umap_result = umap_model.fit_transform(embeddings_array)

    # Add the UMAP results as new columns to the DataFrame
    for i in range(umap_dim):
        df[f'umap_{i+1}'] = umap_result[:, i]

    # Save the updated DataFrame back to a CSV file
    df.to_csv(clusters_path, index=False)

    print("2D embeddings saved to:", clusters_path)