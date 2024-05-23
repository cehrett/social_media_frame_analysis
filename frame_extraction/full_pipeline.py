
import os

import sys
print(sys.path)

# # Config:
# # User-defined inputs
# data_path = os.path.join('/', 'zfs', 'disinfo', 'qatar', 'data', 'split_files', 'part_1.csv')
# output_dir = os.path.join('/', 'zfs', 'disinfo', 'qatar', 'data', 'split_files', 'frame_extraction_analysis')
# text_col = 'tweet'
# api_key_loc = os.path.join('/', 'home', 'cehrett', '.apikeys', 'openai_api_key.txt')
# id_col = 'id'
# prompt_txt_loc =  os.path.join('/', 'zfs', 'disinfo', 'qatar', 'data', 'split_files', 'frame_extraction_analysis', 'system_prompt.txt')
# username='cehrett'

# # Optional user-defined inputs
# # Frame-clustering settings
# umap_dim = 50
# min_cluster_size = 20

# # Suspicious frame-cluster identification settings
# flags = []
# min_num_trials = 5
# use_cuda = True
# device_num = 0

# # Bayesian clustering settings
# needle_var = None
# num_narratives = 16
# component_weights = [10000, 10, 5, 1]
# n_repeats = 40
# epochs = 10
# batch_size = 256
# num_particles = 1
# dropout_rate = 0.3
# cholesky_rank = 2

# # Time series visualization settings
# num_bins = 12
# query_theories = ['This is sportswashing']
# time_col = 'CreatedTime'
# round_to_nearest = 'H'

# # Which parts of the pipeline to run
# do_process_and_save_posts = True
# do_get_embeddings = True
# do_cluster_embeddings = True
# do_run_analysis = False
# do_bayesian_clustering = False
# do_visualize_frame_cluster_across_time = True

# Convert the above config settings to command-line arguments
def parse_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a day's posts, cluster them, collapse the cluster labels within a day, then collapse them into previous day's, then visualize the week.")
    parser.add_argument("data_path", help="Path to the data file.")
    parser.add_argument("output_dir", help="Path to the output directory where results will be stored.")
    parser.add_argument("--text_col", default='tweet', help="Name of the text column.")
    parser.add_argument("--api_key_loc", default=os.path.expanduser('~/.apikeys/openai_api_key.txt'), help="Location of text file containing OpenAI API key.")
    parser.add_argument("--id_col", default='id', help="Name of the id column.")
    parser.add_argument("--prompt_txt_loc", required=True, help="Location of the system prompt text file.")
    parser.add_argument("--username", required=True, help="Username for access control.")

    # Frame-clustering settings
    parser.add_argument("--umap_dim", default=50, help="Number of dimensions for UMAP.")
    parser.add_argument("--min_cluster_size", default=20, help="Minimum cluster size for HDBSCAN.")

    # Suspicious frame-cluster identification settings
    parser.add_argument("--flags", default=[], help="List of flags to use for suspicious frame-cluster identification.")
    parser.add_argument("--min_num_trials", default=5, help="Minimum number of trials to run for suspicious frame-cluster identification.")
    parser.add_argument("--use_cuda", default=True, help="Whether to use CUDA for suspicious frame-cluster identification.")
    parser.add_argument("--device_num", default=0, help="CUDA device number to use for suspicious frame-cluster identification.")

    # Bayesian clustering settings
    parser.add_argument("--needle_var", default=None, help="Needle variance for Bayesian clustering.")
    parser.add_argument("--num_narratives", default=16, help="Number of narratives to identify using Bayesian clustering.")
    parser.add_argument("--component_weights", default=[10000, 10, 5, 1], help="Weights for the components of the Bayesian clustering model.")
    parser.add_argument("--n_repeats", default=40, help="Number of repeats for Bayesian clustering.")
    parser.add_argument("--epochs", default=10, help="Number of epochs for Bayesian clustering.")
    parser.add_argument("--batch_size", default=256, help="Batch size for Bayesian clustering.")
    parser.add_argument("--num_particles", default=1, help="Number of particles for Bayesian clustering.")
    parser.add_argument("--dropout_rate", default=0.3, help="Dropout rate for Bayesian clustering.")
    parser.add_argument("--cholesky_rank", default=2, help="Cholesky rank for Bayesian clustering.")

    # Time series visualization settings
    parser.add_argument("--num_bins", default=12, help="Number of time bins per day.")
    parser.add_argument("--query_theories", default=['This is sportswashing'], nargs='+', help="List of query theories for time series visualization.")
    parser.add_argument("--time_col", default='CreatedTime', help="Column name for the time data.")
    parser.add_argument("--round_to_nearest", default='H', help="Rounding granularity for time data.")
    parser.add_argument("--bin_times", action='store_true', help="Whether to bin the times.")

    # Which parts of the pipeline to run
    parser.add_argument("--do_process_and_save_posts", action='store_true', help="Extract frames from a day's posts.")
    parser.add_argument("--do_get_embeddings", action='store_true', help="Get embeddings for the frames.")
    parser.add_argument("--do_cluster_embeddings", action='store_true', help="Cluster the embeddings of the frames.")
    parser.add_argument("--do_run_analysis", action='store_true', help="Run analysis to identify suspicious frame clusters.")
    parser.add_argument("--do_bayesian_clustering", action='store_true', help="Run Bayesian clustering to identify suspicious account clusters.")
    parser.add_argument("--do_visualize_frame_cluster_across_time", action='store_true', help="Visualize frame clusters across time.")
    
    return parser.parse_args()


def full_pipeline(data_path,
                  output_dir, 
                  text_col, 
                  api_key_loc, 
                  id_col, 
                  prompt_txt_loc, 
                  username, 
                  umap_dim, 
                  min_cluster_size, 
                  flags, 
                  min_num_trials, 
                  use_cuda, 
                  device_num, 
                  needle_var, 
                  num_narratives, 
                  component_weights, 
                  n_repeats, 
                  epochs, 
                  batch_size, 
                  num_particles, 
                  dropout_rate, 
                  cholesky_rank, 
                  num_bins, 
                  query_theories, 
                  time_col, 
                  round_to_nearest, 
                  bin_times,
                  do_process_and_save_posts, 
                  do_get_embeddings, 
                  do_cluster_embeddings, 
                  do_run_analysis, 
                  do_bayesian_clustering, 
                  do_visualize_frame_cluster_across_time
                  ):
    
    if do_process_and_save_posts:
        from frame_extraction.extract_frames import process_and_save_posts
        process_and_save_posts(data_path,
                            results_dir=output_dir,
                            text_col=text_col,
                            api_key_loc=api_key_loc,
                            system_prompt_loc=prompt_txt_loc,
                            labeled_data_path='~/Projects/Trolls/narr_extraction/frame_extraction_tools/data/labeled_data.csv'
                            )

    if do_get_embeddings:
        from frame_extraction.get_frame_embeddings import get_embeddings
        get_embeddings(embeddings_path=os.path.join(output_dir, 'frame_embeddings.json'),
                    frames_path=os.path.join(output_dir, 'frame_extraction_results.csv'),
                    api_key_loc=api_key_loc)

    if do_cluster_embeddings:
        from frame_extraction.cluster_frames import cluster_embeddings
        cluster_embeddings(frames_path=os.path.join(output_dir, 'frame_extraction_results.csv'),
                        embeddings_path=os.path.join(output_dir, 'frame_embeddings.json'),
                        clusters_path=os.path.join(output_dir, 'frame_cluster_results.csv'),
                        umap_dim=umap_dim,
                        min_cluster_size=min_cluster_size,
                        id_col=id_col)

    if do_run_analysis:
        from frame_extraction.find_suspicious_frame_clusters import run_analysis
        run_analysis(cluster_label_loc=os.path.join(output_dir, 'frame_cluster_results.csv'), 
                    flags_loc=data_path, 
                    author_id_loc=data_path, 
                    output_dir=output_dir, 
                    flags=flags, 
                    min_num_trials=min_num_trials, 
                    use_cuda=use_cuda, 
                    device_num=device_num)

    if do_bayesian_clustering:
        from frame_extraction.find_suspicious_account_clusters import bayesian_clustering
        bayesian_clustering(needle_var, 
                            authors_path=os.path.join(output_dir, 'flags_and_num_posts_per_author.csv'), 
                            successes_path=os.path.join(output_dir, 'frame_cluster_use_per_author.csv'), 
                            narratives_path=os.path.join(output_dir, 'top_suspicious_frame_clusters.csv'), 
                            num_narratives=num_narratives, 
                            flags=flags, 
                            component_weights=component_weights, 
                            n_repeats=n_repeats, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            num_particles=num_particles, 
                            dropout_rate=dropout_rate, 
                            cholesky_rank=cholesky_rank)

    if do_visualize_frame_cluster_across_time:
        from frame_extraction.visualize_frame_clusters_across_time import visualize_frame_cluster_across_time
        visualize_frame_cluster_across_time(frame_cluster_results_loc=os.path.join(output_dir, 'frame_cluster_results.csv'),
                    original_data_loc=data_path,
                    frame_cluster_embeddings_loc=os.path.join(output_dir, 'frame_embeddings.json'),
                    num_bins=num_bins,
                    round_to_nearest=round_to_nearest,
                    time_col=time_col,
                    id_col=id_col,
                    num_fcs_to_display=8,
                    figures_output_loc=os.path.join(output_dir,'frame_cluster_activity_across_time.html'),
                    username=username,
                    api_key_loc=api_key_loc,
                    query_theories=query_theories,
                    bin_times=bin_times
                    )

    
if __name__ == "__main__":
    args = parse_command_line_arguments()
    full_pipeline(args.data_path,
                  args.output_dir, 
                  args.text_col, 
                  args.api_key_loc, 
                  args.id_col, 
                  args.prompt_txt_loc, 
                  args.username, 
                  args.umap_dim, 
                  args.min_cluster_size, 
                  args.flags, 
                  args.min_num_trials, 
                  args.use_cuda, 
                  args.device_num, 
                  args.needle_var, 
                  args.num_narratives, 
                  args.component_weights, 
                  args.n_repeats, 
                  args.epochs, 
                  args.batch_size, 
                  args.num_particles, 
                  args.dropout_rate, 
                  args.cholesky_rank, 
                  args.num_bins, 
                  args.query_theories, 
                  args.time_col, 
                  args.round_to_nearest, 
                  args.bin_times,
                  args.do_process_and_save_posts, 
                  args.do_get_embeddings, 
                  args.do_cluster_embeddings, 
                  args.do_run_analysis, 
                  args.do_bayesian_clustering, 
                  args.do_visualize_frame_cluster_across_time
                  )