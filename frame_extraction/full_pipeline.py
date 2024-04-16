
import os

# Config:
# User-defined inputs
data_path = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections', 'AfricanNationalCongress', '2024-03-23.csv')
output_dir = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections', 'frame_extraction_analysis', 'outputs', 'AfricanNationalCongress', '2024-03-23')
text_col = 'Message'
api_key_loc = os.path.join('/', 'home', 'cehrett', '.apikeys', 'openai_api_key.txt')
id_col = 'UniversalMessageId'
prompt_txt_loc = os.path.join('/', 'zfs', 'disinfo', 'hunts', 'Africa_Elections', 'frame_extraction_analysis', 'system_prompt.txt')
username='cehrett'

# Optional user-defined inputs
# Frame-clustering settings
umap_dim = 50
min_cluster_size = 10

# Suspicious frame-cluster identification settings
flags = []
min_num_trials = 5
use_cuda = True
device_num = 0

# Bayesian clustering settings
needle_var = None
num_narratives = 16
component_weights = [10000, 10, 5, 1]
n_repeats = 40
epochs = 10
batch_size = 256
num_particles = 1
dropout_rate = 0.3
cholesky_rank = 2

# Time series visualization settings
num_bins = 12
query_theories = ['There is corruption in the African National Congress', 
                  'The Zondo Commission findings are a threat to the ANC', 
                  'The Judicial Commission of Inquiry into Allegations of State Capture must be taken seriously']
time_col = 'CreatedTime'
round_to_nearest = 'H'

# Which parts of the pipeline to run
do_process_and_save_tweets = False
do_get_embeddings = True
do_cluster_embeddings = True
do_run_analysis = False
do_bayesian_clustering = False
do_visualize_frame_cluster_across_time = True

if do_process_and_save_tweets:
    from extract_frames import process_and_save_tweets
    process_and_save_tweets(data_path,
                           results_dir=output_dir,
                           text_col=text_col,
                           api_key_loc=api_key_loc,
                           system_prompt_loc=prompt_txt_loc
                          )

if do_get_embeddings:
    from get_frame_embeddings import get_embeddings
    get_embeddings(embeddings_path=os.path.join(output_dir, 'frame_embeddings.json'),
                  frames_path=os.path.join(output_dir, 'frame_extraction_results.csv'),
                  api_key_loc=api_key_loc)

if do_cluster_embeddings:
    from cluster_frames import cluster_embeddings
    cluster_embeddings(frames_path=os.path.join(output_dir, 'frame_extraction_results.csv'),
                      embeddings_path=os.path.join(output_dir, 'frame_embeddings.json'),
                      clusters_path=os.path.join(output_dir, 'frame_cluster_results.csv'),
                      umap_dim=umap_dim,
                      min_cluster_size=min_cluster_size,
                      id_col=id_col)

if do_run_analysis:
    from find_suspicious_frame_clusters import run_analysis
    run_analysis(cluster_label_loc=os.path.join(output_dir, 'frame_cluster_results.csv'), 
                 flags_loc=data_path, 
                 author_id_loc=data_path, 
                 output_dir=output_dir, 
                 flags=flags, 
                 min_num_trials=min_num_trials, 
                 use_cuda=use_cuda, 
                 device_num=device_num)

if do_bayesian_clustering:
    from find_suspicious_account_clusters import bayesian_clustering
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
    from visualize_frame_clusters_across_time import visualize_frame_cluster_across_time
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
                 query_theories=query_theories
                )

    
