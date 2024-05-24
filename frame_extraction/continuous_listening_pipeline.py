# Imports
from .extract_frames import process_and_save_posts
from .get_frame_embeddings import get_embeddings
from .cluster_frames import cluster_embeddings
from .get_multiday_visualization import get_single_and_multiday_visualizations
from .collapse_cluster_labels import collapse
from .get_cluster_description import get_cluster_descriptions
from .utils.load_llm_model import prepare_to_load_model
from .utils.make_table_from_store import make_table

import os

# Get file path from topic and date
def get_file_path(root_dir, topic, date):
    """
    Get the file path for a given topic and date.
    """
    return os.path.join(root_dir, topic, date + '.csv')


# Get directory in which to store results
def get_results_dir(root_dir, topic, date=None ):
    """
    Get the directory in which to store results.
    """
    if date is None:
        return os.path.join(root_dir, 'frame_extraction_analysis', 'outputs', topic)
    else:
        return os.path.join(root_dir, 'frame_extraction_analysis', 'outputs', topic, date)
    

# Process command-line arguments
def process_command_line_args():
    """
    Process command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a day's posts, cluster them, \
    collapse the cluster labels within a day, then collapse them into previous day's, then visualize the week.")
    parser.add_argument("root_dir", 
                        help="Root directory for the project. This should be the parent dir of which the topic dirs are children."
                             "Each topic dir should have csvs: YYYY-MM-DD.csv")
    parser.add_argument("--topic", required=True, help="Topic for the analysis.")
    parser.add_argument("--date", required=True, help="Date for the analysis.")
    parser.add_argument("--system_prompt_loc", required=True, help="System prompt to give to LLM when extracting frames.")

    parser.add_argument("--labeled_data_path", default='/zfs/disinfo/narratives/labeled_data.csv', 
        help="Path to labeled data CSV. Defaults to '/zfs/disinfo/narratives/labeled_data.csv'.")
    parser.add_argument("--text_col", default='Message', help="Name of the text column.")
    parser.add_argument("--api_key_loc", default='./openai_api_key.txt', help="Location of text file containing OpenAI API key.")
    parser.add_argument("--raw_csv_or_intermediate", default='c', 
        help="Whether to use the input path data file implied by the 'root_dir', \
        'topic' and 'date' inputs (c), or an intermediate file (i). Default (c). \
        Only use (i) if previous frame extraction was interrupted before completion.")

    parser.add_argument("--umap_dim", default=50, help="Number of dimensions for UMAP.")
    parser.add_argument("--min_cluster_size", default=10, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument("--id_col", default='UniversalMessageId', help="Name of the id column.")

    parser.add_argument("--num_bins", default=12, help="Number of time bins per day.")
    parser.add_argument("--round_to_nearest", default='H', help="Rounding granularity for time data.")
    parser.add_argument("--time_col", default='CreatedTime', help="Column name for the time data.")
    parser.add_argument("--num_fcs_to_display", default=8, help="Number of frame clusters to display.")
    parser.add_argument("--username", required=True, help="Username for access control.")
    parser.add_argument("--query_theories", required=True, help="Semicolon-separated list of query theories.")

    parser.add_argument("--website_dir", default='/zfs/disinfo/website', help="Directory for the website.")

    parser.add_argument("--extract_frames", action='store_true', help="Extract frames.")
    parser.add_argument("--get_embeddings", action='store_true', help="Get embeddings.")
    parser.add_argument("--cluster_embeddings", action='store_true', help="Cluster embeddings.")
    parser.add_argument("--get_descriptions", action='store_true', help="Get cluster descriptions.")
    parser.add_argument("--collapse_within_day", action='store_true', help="Collapse cluster labels within-day.")
    parser.add_argument("--collapse_into_store", action='store_true', help="Collapse cluster labels into frame store.")
    parser.add_argument("--visualize", action='store_true', help="Visualize frame clusters across time.")
    parser.add_argument("--add_result_to_website", action='store_true', help="Add result to website.")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = process_command_line_args()

    # Get file path and results directory
    original_data_loc = get_file_path(args.root_dir, args.topic, args.date)
    results_dir = get_results_dir(args.root_dir, args.topic, args.date)
    store_dir = get_results_dir(args.root_dir, args.topic)

    # If extracting frames, getting embeddings, getting descriptions, or collapsing, add API key to environment
    if args.extract_frames or args.get_embeddings or args.get_descriptions or args.collapse_within_day or args.collapse_into_store:
        prepare_to_load_model(api_key_loc=args.api_key_loc)

    # Check whether the results directory exists; if not, create it
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    # Extract frames, get embeddings, cluster embeddings, and visualize frame clusters across time
    if args.extract_frames:
        print("Extracting frames...")
        process_and_save_posts(input_path=original_data_loc,
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

    if args.get_descriptions:
        print("Getting cluster descriptions...")
        get_cluster_descriptions(input_file=os.path.join(results_dir, 'frame_cluster_results.csv'),
                                output_file=os.path.join(results_dir, 'frame_cluster_results.csv'),
                                api_key_loc=args.api_key_loc,
                                n_samp=10,
                                model='gpt-4o'
                                )    


    if args.collapse_within_day:
        print("Collapsing cluster labels within-day...")
        collapse(root_dir=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                topic=args.topic,
                date_current=args.date,
                model='gpt-4o',
                across_days=False
                )
        
    if args.collapse_into_store:
        print(f"Collapsing cluster labels into frame store for {args.topic}...")
        collapse(root_dir=os.path.join(args.root_dir, 'frame_extraction_analysis', 'outputs'),
                topic=args.topic,
                date_current=args.date,
                model='gpt-4o',
                store_loc='frame_store.csv'
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
        

    if args.add_result_to_website:
        print("Adding result to website...")

        # Check if website dir exists
        if not os.path.exists(args.website_dir):
            raise FileNotFoundError(f"Directory {args.website_dir} does not exist.")
        
        # Use `make_table` to convert the frame store into a table
        store_moveto_loc = os.path.join(args.website_dir, f'{args.topic}.html')
        make_table(clusters_input=os.path.join(store_dir, 'frame_store.csv'),
                   output_file=store_moveto_loc,
                   n_samp=6,
                   topic_dir=args.topic,
                   id_col=args.id_col,
                   text_col=args.text_col)
        
        # Move the file `frame_cluster_activity_across_time.html` in the results_dir to the website directory
        import shutil
        plots_moveto_loc = os.path.join(args.website_dir, args.topic, f'{args.date}.html')
        shutil.copy(os.path.join(results_dir, 'frame_cluster_activity_across_time.html'), 
                    plots_moveto_loc)
        print(f"Moved {os.path.join(results_dir, 'frame_cluster_activity_across_time.html')} to {plots_moveto_loc}.")

        # Recreate the README.md file in the website directory, based on the files present
        # Each subdirectory of args.website is a topic, which gets a ### header. Each file gets a link with the filename (minus .html).
        intro_paragraph = """\
Below is a list of dates for each of several topics. \
For each topic, every day we collect the tweets that reference that topic. \
Then we feed each tweet to an AI pipeline that identifies any and all \
claims of broad social significance expressed in the tweet – which we call “frames”. \
E.g., “Ready to vote to keep Ramaphosa at the helm where he belongs” expresses the frame \
“Ramaphosa should win reelection”, whereas “Whoa, just served Ramaphosa’s son in the Starbucks I work in” \
does not express any frames. Once we have the frames expressed by each tweet, \
we group them together into clusters that all express variants of roughly the same frame. \
Then we track the activity of these clusters over time, to see what frames are being discussed online. \
The link for each date goes to a page with plots showing topic activity for that date, \
as well as for the week preceding that date. \
For each topic, a separate page provides a list of all clusters of frames found for that topic.\n\n"""
        with open(os.path.join(args.website_dir, 'README.md'), 'w') as f:
            f.write("# Social media analysis of May 29 South African elections using LLM-driven frame extraction\n" +
                    "Analysis of social media posts related to the May 29, 2024 South Africa elections.\n\n" +
                    "## Time series of frames expressed in social media posts\n\n" + 
                    intro_paragraph + "\n\n")
            for topic in os.listdir(args.website_dir):
                if os.path.isdir(os.path.join(args.website_dir, topic)) and topic != '.git':
                    f.write(f"\n### {topic} [(Cluster labels)]({topic}.html)\n")
                    for file in os.listdir(os.path.join(args.website_dir, topic)):
                        if file.endswith('.html'):
                            f.write(f"- [{file[:-5]}]({os.path.join(topic, file)})\n")
