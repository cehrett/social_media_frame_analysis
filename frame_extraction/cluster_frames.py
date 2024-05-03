import pandas as pd
import os
import csv
import ast
import string
import json
import traceback
from .utils import clustering_tools as ct
import numpy as np 

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan

def cluster_embeddings(frames_path, embeddings_path, clusters_path, umap_dim=50, min_cluster_size=10, id_col='id'):
    # Load the frames into a dataframe
    df_with_frames = pd.read_csv(frames_path)
    df_with_frames['frames'] = df_with_frames['frames'].apply(safe_literal_eval)
    df_with_frames['frames'] = df_with_frames['frames'].apply(
        lambda x: [elt.strip(string.whitespace + string.punctuation)
                   for elt in x if elt is not None] if isinstance(x, list) else x)
    
    # Handle corner cases by marking them as errors
    df_with_frames.loc[~df_with_frames.frames.apply(lambda x: isinstance(x, list)), 'frames'] = [['ERROR']]
    
    # Load the embeddings
    with open(embeddings_path, 'r') as file:
        embeddings_dict = json.load(file)
    
    # Process embeddings and perform clustering
    embeddings_dict = ct.get_theories_and_embeddings_using_dict(df=df_with_frames, data_dict=embeddings_dict, id_col=id_col)
    df_frame_clusters = pd.DataFrame(embeddings_dict)
    # import pdb; pdb.set_trace()
    print('Beginning clustering')
    try:
        frame_clustering = ct.perform_clustering(df_frame_clusters['embeddings'].tolist(),
                                                 umap_dim_size=umap_dim,
                                                 min_cluster_size=min_cluster_size)
        print('Clustering complete')

        # Update dataframe with cluster labels
        df_frame_clusters['cluster_labels'] = frame_clustering['hdb'].labels_
        
        # Drop embeddings column
        df_frame_clusters.drop('embeddings', axis=1, inplace=True)

        # Save clustering results
        # import pdb; pdb.set_trace()
        df_frame_clusters.to_csv(clusters_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Frame clustering completed successfully. Results stored in {clusters_path}.")

        # Output some metrics
        total_number_of_clusters = df_frame_clusters['cluster_labels'].nunique() - 1
        total_number_of_outliers = sum(df_frame_clusters['cluster_labels'] == -1)
        average_rows_per_cluster = df_frame_clusters[df_frame_clusters['cluster_labels'] != -1].groupby('cluster_labels').size().mean()

        print(f"Total of {total_number_of_clusters} clusters found, with average of {average_rows_per_cluster:.2f} posts per cluster.")
        print(f"{total_number_of_outliers} outliers found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

# Retain CLI functionality
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster embeddings of frames.")
    parser.add_argument("--frames_path", 
                        default=os.path.join('.', 'output', 'frame_extraction_results.csv'), 
                        help="Path of csv file where frames are stored; output of extract_frames.py.")
    parser.add_argument("--embeddings_path",
                        default=os.path.join('.', 'data', 'frame_embeddings.json'),
                        help="Path of json file where frame embeddings are stored; output of get_frame_embeddings.py.")
    parser.add_argument("--clusters_path",
                        default=os.path.join('.', 'output', 'frame_cluster_results.csv'), 
                        help="Path where clustering results will be stored as a csv.")
    parser.add_argument("--umap_dim",
                        type=int,
                        default=50, 
                        help="Dimension to which to UMAP-reduce the embeddings prior to using HDBScan. Default 50.")
    parser.add_argument("--min_cluster_size",
                        type=int,
                        default=10, 
                        help="Controls the min_cluster_size parameter of HDBScan. Default 10.")
    parser.add_argument("--id_col",
                        default='id',
                        help="Unique post id column in frames dataset. Default 'id'.")
    
    args = parser.parse_args()
    
    cluster_embeddings(frames_path=args.frames_path,
                       embeddings_path=args.embeddings_path,
                       clusters_path=args.clusters_path,
                       umap_dim=args.umap_dim,
                       min_cluster_size=args.min_cluster_size,
                       id_col=args.id_col)
