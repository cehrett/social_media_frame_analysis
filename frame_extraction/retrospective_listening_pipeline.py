# Imports
from .extract_frames import process_and_save_posts
from .get_frame_embeddings import get_embeddings
from .cluster_frames import cluster_embeddings
from .get_cluster_description import get_cluster_descriptions
from .utils.load_llm_model import prepare_to_load_model
from .visualize_frame_clusters_across_time import visualize_frame_cluster_across_time
from .utils.get_2d_embeddings import get_2d_embeddings

import os
import argparse

# Process command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Retrospective Listening Pipeline')
    
    parser.add_argument('--num_clusters', type=int, default=20, help='Number of clusters to generate')
    return parser.parse_args()


# Process command-line arguments
def process_command_line_args():
    """
    Process command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a day's posts, cluster them, visualize them.")

    # Data location arguments
    parser.add_argument('--data_loc', type=str, default='data.csv', help='Path to data file')
    parser.add_argument('--output_path', type=str, default='output/', help='Path to output directory, where results will be saved')
    parser.add_argument("--system_prompt_loc", required=True, help="System prompt to give to LLM when extracting frames.")

    # Frame extraction arguments
    parser.add_argument("--labeled_data_path", required=True, 
        help="Path to labeled data CSV. Should be a csv with a column for post text and a column for frames.")
    parser.add_argument("--text_col", default='Message', help="Name of the text column in the data file.")
    parser.add_argument("--api_key_loc", default='./openai_api_key.txt', help="Location of text file containing OpenAI API key.")
    parser.add_argument("--raw_csv_or_intermediate", default='c', 
        help="Whether to extract frames from the original data file, (c), or an intermediate file (i). Default (c). \
        Only use (i) if previous frame extraction was interrupted before completion.")

    # Frame clustering arguments
    parser.add_argument("--umap_dim", type=int, default=50, help="Number of dimensions for UMAP.")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument("--id_col", type=str, default='UniversalMessageId', help="Name of the id column in the data file.")

    # Suspicious cluster identification arguments
    parser.add_argument("--flags", default=[], nargs='+',
                        help="List of flags to use for suspicious cluster identification. Headers of flag cols in original data.")
    
    # Bayesian clustering arguments
    parser.add_argument("--needle_var", type=str, default='is_troll', 
                        help="Needle indicator for clustering. Header of col which is 1 for known trolls, 0 otherwise.")
    parser.add_argument("--num_narratives", type=int, default=16, help="Number of narratives to identify using the clustering algorithm.")
    parser.add_argument("--component_weights", default=[10000, 10, 5, 1], help="Weights for the components of the Bayesian clusters.")
    parser.add_argument("--n_repeats", type=int, default=40, help="Number of repeats for the Bayesian clustering.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for the Bayesian clustering.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the Bayesian clustering.")
    parser.add_argument("--num_particles", type=int, default=1, help="Number of particles for the Bayesian clustering.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for the Bayesian clustering.")
    parser.add_argument("--cholesky_rank", type=int, default=2, help="Cholesky rank for the Bayesian clustering.")

    # Visualization arguments
    parser.add_argument("--num_bins", type=int, default=12, help="Number of time bins into which to divide the data, for visualization.")
    parser.add_argument("--round_to_nearest", type=str, default='H', help="Rounding granularity for time data. Default 'H' is hourly.")
    parser.add_argument("--time_col", type=str, default='CreatedTime', help="Column name for the time data in the data file.")
    parser.add_argument("--num_fcs_to_display", type=int, default=8, help="Number of frame clusters to display in the visualization.")
    parser.add_argument("--query_theories", default=['There is a conspiracy'], help="Semicolon-separated list of queries, \
                        used to get frame-clusters relevant to the queries. These frame-clusters are included in the visualization.")
    
    # Main arguments: which parts of the pipeline to run
    parser.add_argument("--extract_frames", action='store_true', help="Extract frames.")
    parser.add_argument("--get_embeddings", action='store_true', help="Get embeddings.")
    parser.add_argument("--cluster_embeddings", action='store_true', help="Cluster embeddings.")
    parser.add_argument("--get_descriptions", action='store_true', help="Get cluster descriptions.")
    parser.add_argument("--get_2d_embeddings", action='store_true', help="Get 2D embeddings.")
    parser.add_argument("--get_suspicious_clusters", action='store_true', help="Get suspicious clusters.")
    parser.add_argument("--get_account_clusters", action='store_true', help="Get account clusters.")
    parser.add_argument("--visualize", action='store_true', help="Visualize frame clusters across time.")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("\nRetrospective Listening Pipeline\n")
    args = process_command_line_args()

    # If extracting frames, getting embeddings, getting descriptions, or collapsing, add API key to environment
    if args.extract_frames or args.get_embeddings or args.get_descriptions:
        prepare_to_load_model(api_key_loc=args.api_key_loc)

    # Check whether the results directory exists; if not, create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"Created directory: {args.output_path}")
    
    if args.extract_frames:
        print("\nExtracting frames...")
        process_and_save_posts(
            input_path=args.data_loc, 
            results_dir=args.output_path, 
            system_prompt_loc=args.system_prompt_loc, 
            labeled_data_path=args.labeled_data_path, 
            text_col=args.text_col, 
            api_key_loc=args.api_key_loc, 
            raw_csv_or_intermediate=args.raw_csv_or_intermediate, 
        )

    if args.get_embeddings:
        print("\nGetting embeddings...")
        get_embeddings(
            embeddings_path=os.path.join(args.output_path, 'frame_embeddings.json'),
            frames_path=os.path.join(args.output_path, 'frame_extraction_results.csv'),
            api_key_loc=args.api_key_loc,
        )
        
    if args.cluster_embeddings:
        print("\nClustering embeddings...")
        cluster_embeddings(
            frames_path=os.path.join(args.output_path, 'frame_extraction_results.csv'),
            embeddings_path=os.path.join(args.output_path, 'frame_embeddings.json'),
            clusters_path=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            umap_dim=args.umap_dim,
            min_cluster_size=args.min_cluster_size,
            id_col=args.id_col,
        )

    if args.get_descriptions:
        print("\nGetting cluster descriptions...")
        get_cluster_descriptions(
            input_file=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            output_file=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            api_key_loc=args.api_key_loc,
            n_samp=10,
            model='gpt-4o',
        )

    if args.get_2d_embeddings:
        print("\nGetting 2D embeddings...")
        get_2d_embeddings(
            embeddings_path=os.path.join(args.output_path, 'frame_embeddings.json'),
            frames_path=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            umap_dim=2,
        )

    if args.get_suspicious_clusters:
        from frame_extraction.find_suspicious_frame_clusters import run_analysis
        run_analysis(
            cluster_label_loc=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            flags_loc=args.data_loc,
            flags=args.flags,
            output_dir=args.output_path,
            use_cuda=True,
            device_num=0,            
        )

    if args.get_account_clusters:
        from frame_extraction.find_suspicious_account_clusters import bayesian_clustering
        bayesian_clustering(needle_var=args.needle_var,
                            authors_path=os.path.join(args.output_path, 'flags_and_num_posts_per_author.csv'),
                            successes_path=os.path.join(args.output_path, 'frame_cluster_use_per_author.csv'),
                            narratives_path=os.path.join(args.output_path, 'top_suspicious_frame_clusters.csv'),
                            num_narratives=args.num_narratives,
                            flags=args.flags,
                            component_weights=args.component_weights,
                            n_repeats=args.n_repeats,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            num_particles=args.num_particles,
                            dropout_rate=args.dropout_rate,
                            cholesky_rank=args.cholesky_rank,
        )

    if args.visualize:
        print("\nVisualizing frame clusters across time...")
        visualize_frame_cluster_across_time(
            frame_cluster_results_loc=os.path.join(args.output_path, 'frame_cluster_results.csv'),
            original_data_loc=args.data_loc,
            frame_cluster_embeddings_loc=os.path.join(args.output_path, 'frame_embeddings.json'),
            num_bins=args.num_bins,
            round_to_nearest=args.round_to_nearest,
            time_col=args.time_col,
            id_col=args.id_col,
            num_fcs_to_display=args.num_fcs_to_display,
            figures_output_loc=os.path.join(args.output_path,'frame_clusters_across_time.html'),
            username=None,
            api_key_loc=args.api_key_loc,
            query_theories=args.query_theories,
            multiday=False,
            return_figures=False,
        )