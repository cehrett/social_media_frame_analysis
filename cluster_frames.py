import pandas as pd
import os
import argparse
import csv
import ast
import string
import json
import traceback
import utils.clustering_tools as ct


def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings of frames.")
    parser.add_argument("--frames_path", 
                        default=os.path.join('.', 'frame_extraction_results.csv'), 
                        help="Path of csv file where frames are stored; output of extract_frames.py.")
    parser.add_argument("--embeddings_path",
                        default=os.path.join('.', 'data', 'frame_embeddings.json'),
                        help="Path of json file where frame embeddings are stored; output of get_frame_embeddings.py.")
    parser.add_argument("--clusters_path",
                        default=os.path.join('.', 'frame_cluster_results.csv'), 
                        help="Path where clustering results will be stored as a csv.")
    parser.add_argument("--umap_dim",
                        type=int,
                        default=50, 
                        help="Dimension to which to UMAP-reduce the embeddings prior to using HDBScan.")
    parser.add_argument("--min_cluster_size",
                        type=int,
                        default=10, 
                        help="Controls the min_cluster_size parameter of HDBScan.")
    
    args = parser.parse_args()
    
    # Load the frames into a dataframe
    df_with_frames = pd.read_csv(args.frames_path)
    df_with_frames.frames = df_with_frames.frames.apply(safe_literal_eval)
    df_with_frames.frames = \
        df_with_frames.frames.apply(
            lambda x: [elt.strip(string.whitespace + string.punctuation) \
                for elt in x if elt is not None] if isinstance(x, list) else x)
    
    # Load the embeddings
    with open(args.embeddings_path, 'r') as file:
        embeddings_dict = json.load(file)
    
    # Get embeddings for each theory in the data, with duplicates
    embeddings_dict = ct.get_theories_and_embeddings_using_dict(df=df_with_frames, data_dict=embeddings_dict, id_col='id')
    df_frame_clusters = pd.DataFrame(embeddings_dict)

    # Get clustering results
    print(f'Beginning clustering')
    try:
        frame_clustering = ct.perform_clustering(df_frame_clusters.embeddings.tolist(),
                                                 umap_dim_size=args.umap_dim,
                                                 min_cluster_size=args.min_cluster_size
                                                )
        print(f'Clustering complete')

        # Put them in the df
        df_frame_clusters['cluster_labels'] = frame_clustering['hdb'].labels_

        # Save results of clustering
        df_frame_clusters.to_csv(args.clusters_path, index=False, quoting=csv.QUOTE_ALL)
        
        print(f"Frame clustering completed successfully. Results stored in {args.clusters_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        # import pdb; pdb.post_mortem()
        
if __name__ == "__main__":
    main()