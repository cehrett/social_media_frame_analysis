# Imports
from .extract_frames import process_and_save_tweets
from .get_frame_embeddings import get_embeddings
from .cluster_frames import cluster_embeddings
from .get_multiday_visualization import get_single_and_multiday_visualizations
from .collapse_cluster_labels import collapse

import os

# Get file path from topic and date
def get_file_path(root_dir, topic, date):
    """
    Get the file path for a given topic and date.
    """
    return os.path.join(root_dir, topic, date + '.csv')


# Get directory in which to store results
def get_results_dir(root_dir, topic, date, ):
    """
    Get the directory in which to store results.
    """
    return os.path.join(root_dir, 'frame_extraction_analysis', 'outputs', topic, date)
    

# Process command-line arguments
def process_command_line_args():
    """
    Process command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a day's posts, cluster them, collapse the cluster labels within a day, then collapse them into previous day's, then visualize the week.")
    parser.add_argument("root_dir", help="Root directory for the project.")
    parser.add_argument("--topic", required=True, help="Topic for the analysis.")
    parser.add_argument("--date", required=True, help="Date for the analysis.")
    parser.add_argument("--system_prompt_loc", required=True, help="System prompt to give to LLM when extracting frames.")

    parser.add_argument("--labeled_data_path", default='/zfs/disinfo/narratives/labeled_data.csv', help="Path to labeled data CSV. Defaults to '/zfs/disinfo/narratives/labeled_data.csv'.")
    parser.add_argument("--text_col", default='Message', help="Name of the text column.")
    parser.add_argument("--api_key_loc", default='./openai_api_key.txt', help="Location of text file containing OpenAI API key.")
    parser.add_argument("--raw_csv_or_intermediate", default='c', help="Whether to use the input path data file implied by the 'root_dir', 'topic' and 'date' inputs (c), or an intermediate file (i). Default (c). Only use (i) if previous frame extraction was interrupted before completion.")

    parser.add_argument("--umap_dim", default=50, help="Number of dimensions for UMAP.")
    parser.add_argument("--min_cluster_size", default=10, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument("--id_col", default='UniversalMessageId', help="Name of the id column.")

    parser.add_argument("--num_bins", default=12, help="Number of time bins per day.")
    parser.add_argument("--round_to_nearest", default='H', help="Rounding granularity for time data.")
    parser.add_argument("--time_col", default='CreatedTime', help="Column name for the time data.")
    parser.add_argument("--num_fcs_to_display", default=8, help="Number of frame clusters to display.")
    parser.add_argument("--username", required=True, help="Username for access control.")
    parser.add_argument("--query_theories", required=True, help="Semicolon-separated list of query theories.")

    parser.add_argument("--extract_frames", action='store_true', help="Extract frames.")
    parser.add_argument("--get_embeddings", action='store_true', help="Get embeddings.")
    parser.add_argument("--cluster_embeddings", action='store_true', help="Cluster embeddings.")
    parser.add_argument("--collapse", action='store_true', help="Collapse cluster labels within-day and across days.")
    parser.add_argument("--visualize", action='store_true', help="Visualize frame clusters across time.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = process_command_line_args()

    # Get file path and results directory
    original_data_loc = get_file_path(args.root_dir, args.topic, args.date)
    results_dir = get_results_dir(args.root_dir, args.topic, args.date)

    # Check whether the results directory exists; if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # Extract frames, get embeddings, cluster embeddings, and visualize frame clusters across time
    if args.extract_frames:
        print("Extracting frames...")
        process_and_save_tweets(input_path=original_data_loc,
                                results_dir=results_dir,
                                labeled_data_path=args.labeled_data_path,
                                text_col=args.text_col,
                                api_key_loc=args.api_key_loc,
                                raw_csv_or_intermediate=args.raw_csv_or_intermediate,
                                system_prompt_loc=args.system_prompt_loc
                            )
    
    if args.get_embeddings:
        print("Getting embeddings...")
        get_embeddings(embeddings_path=os.path.join(results_dir, 'frame_embeddings.json'),
                    frames_path=os.path.join(results_dir, 'frame_extraction_results.csv'),
                    api_key_loc=args.api_key_loc
                    )
    
    if args.cluster_embeddings:
        print("Clustering embeddings...")
        cluster_embeddings(frames_path=os.path.join(results_dir, 'frame_extraction_results.csv'),
                        embeddings_path=os.path.join(results_dir, 'frame_embeddings.json'),
                        clusters_path=os.path.join(results_dir, 'frame_cluster_results.csv'),
                        umap_dim=args.umap_dim,
                        min_cluster_size=args.min_cluster_size,
                        id_col=args.id_col
                        )
        
    if args.collapse:
        print("Collapsing cluster labels within-day...")
        collapse(root_dir=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                topic=args.topic,
                date_current=args.date,
                model='gpt-4-turbo-preview',
                across_days=False
                )
        
        print("Collapsing cluster labels across days...")
        collapse(root_dir=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                topic=args.topic,
                date_current=args.date,
                model='gpt-4-turbo-preview',
                across_days=True
                )
    
    if args.visualize:
        print("Visualizing frame clusters across time...")
        get_single_and_multiday_visualizations(frame_cluster_results_loc=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                                            original_data_loc=original_data_loc,
                                            frame_cluster_embeddings_loc=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                                            num_bins=args.num_bins,
                                            round_to_nearest=args.round_to_nearest,
                                            time_col=args.time_col,
                                            id_col=args.id_col,
                                            num_fcs_to_display=args.num_fcs_to_display,
                                            figures_output_loc=results_dir,
                                            username=args.username,
                                            api_key_loc=args.api_key_loc,
                                            query_theories=args.query_theories.split(';'),
                                            topic=args.topic,
                                            last_day=args.date
                                            )