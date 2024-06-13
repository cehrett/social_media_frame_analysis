import os
import sys
import pandas as pd

from .utils.load_llm_model import prepare_to_load_model
from .extract_frames import process_and_save_posts
from .get_frame_embeddings import get_embeddings
from .cluster_frames import cluster_embeddings


'''
Function that acts as a simplified pipeline function which takes in a dataframe and API key location,
and returns the dataframe with frames and cluster label
'''

def apply_simplified_pipeline(df, api_key_loc='./openai_api_key.txt'):

    # Add unique ID values to dataframe
    df['id'] = range(len(df))

    print(df.head())

    prepare_to_load_model(api_key_loc=api_key_loc)

    # Get directory from which file is being run
    cwd = os.getcwd()

    # Get the directory containing the current file
    current_file_path = os.path.realpath(__file__)
    extraction_directory = os.path.dirname(current_file_path)

    df.to_csv(os.path.join(cwd, 'temp_df.csv'), index=False)

    path_to_csv = os.path.join(cwd, 'temp_df.csv')

    # Ensure labeled data path exist for few-shot learning
    if not os.path.exists(os.path.join(cwd,'labeled_data.csv')):
        raise FileNotFoundError("File 'labeled_data.csv' not found in current directory.")
    

    print("Extracting frames...")
    process_and_save_posts(input_path=path_to_csv,
                                results_dir=cwd,
                                labeled_data_path=os.path.join(cwd,'labeled_data.csv'),
                                text_col=str(list(df.columns)[0]),
                                api_key_loc=api_key_loc,
                                raw_csv_or_intermediate='c',
                                system_prompt_loc= os.path.join(extraction_directory, 'utils/oai_system_message_template.txt')
                            )
    
    # Take the newly created frame_extraction_results.csv file and append frames to df
    frames_df = pd.read_csv(os.path.join(cwd, 'frame_extraction_results.csv'))
    
    df['frames'] = frames_df['frames']

    print("Getting embeddings...")
    get_embeddings(embeddings_path=os.path.join(cwd, 'frame_embeddings.json'),
                frames_path=os.path.join(cwd, 'frame_extraction_results.csv'),
                api_key_loc=api_key_loc
                )
    
    print("Clustering embeddings...")
    cluster_embeddings(frames_path=os.path.join(cwd, 'frame_extraction_results.csv'),
                    embeddings_path=os.path.join(cwd, 'frame_embeddings.json'),
                    clusters_path=os.path.join(cwd, 'frame_cluster_results.csv'),
                    umap_dim=50,
                    min_cluster_size=10,
                    id_col="id"
                    )
    
    # Add the cluster assignments to the original dataframe
    cluster_df = pd.read_csv(os.path.join(cwd, 'frame_cluster_results.csv'))
    df['cluster_labels'] = cluster_df['cluster_labels']

    # Finally drop id column
    df = df.drop(columns=['id'])

    # Clean created files
    os.remove(os.path.join(cwd,'temp_df.csv'))
    os.remove(os.path.join(cwd, 'frame_extraction_results.csv'))
    os.remove(os.path.join(cwd, 'frame_embeddings.json'))
    os.remove(os.path.join(cwd, 'frame_cluster_results.csv'))

    return df

    

    
