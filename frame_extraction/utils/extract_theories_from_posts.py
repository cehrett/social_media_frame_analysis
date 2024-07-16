# Imports
from .load_llm_model import prepare_to_load_model, load_oai_model
from .make_prompt_narr_extraction import make_prompt_for_oai_narr_extr, load_labeled_examples
from .postproc_narr_results import convert_to_series
# Import other libraries
import pandas as pd
import os
import ast
from tqdm import tqdm
import argparse 
from IPython.display import clear_output
import time
import re
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Process some files.")
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('-i', '--input_path', required=True, help="Path to the input CSV or Excel directory.")
    parser.add_argument('-t', '--post_col', required=True, help="Name of the column containing post text.")
    parser.add_argument('-int', '--intermediate_path', required=True, help="Path to save intermediate results CSV.")
    parser.add_argument('-o', '--output', required=True, help="Path to save final results CSV.")
    parser.add_argument('-l', '--labeled_data_path', default='/zfs/disinfo/narratives/labeled_data.csv', help="Path to labeled data CSV. Defaults to '/zfs/disinfo/narratives/labeled_data.csv'.")
    args = parser.parse_args()
    return args

def get_results_from_row(row, 
                         post_col, 
                         model, 
                         labeled_df,
                         system_prompt,
                         verbose=False
                        ):
    # Define helper function that will extract theory from a row of the df.
    # This function will use a random sampling of the labeled data to create a FSL prompt each time it extracts theories from a post.
    # Get the post from this row
    text = row[post_col]

    # Make a prompt using randomly sampled labeled data
    random_labeled_data = load_labeled_examples(labeled_df)
    prompt = make_prompt_for_oai_narr_extr(text, random_labeled_data, system_template=system_prompt)

    try:
        results = model.invoke(prompt)
        narrs = ast.literal_eval(results.content)
    except:
        narrs = ['ERROR']

    if verbose:
        clear_output(wait=True)
        print('POST: ', text, '\n', 'FRAME: ', narrs)
    time.sleep(0.01) 
    return(narrs)

def process_posts(input_path, 
                   post_col, 
                   intermediate_path, 
                   output_path, 
                   labeled_data_path='./data/labeled_data.csv',
                   model_id='gpt-3.5-turbo',
                   api_key_loc='./openai_api_key.txt',
                   chunk_size=100,
                   raw_csv_or_intermediate=None,
                   system_prompt_loc='.utils/oai_system_message_template.txt'
                  ):
    
    # Initialize tqdm with pandas 
    tqdm.pandas()

    # Set input parameters for model settings
    model_id = model_id 
    username = os.environ.get('USER')
    temperature = 1 
    top_p = 0.95
    max_new_tokens = 300
    num_return_sequences = 5
    do_sample=True

    prepare_to_load_model(username, 
                          service='openai', 
                          api_key_loc=api_key_loc)

    # Load the model, as a hugging face pipeline object
    model = load_oai_model(model_id, 
                           temperature=temperature, 
                           top_p=top_p, 
                           max_new_tokens=max_new_tokens, 
                           num_return_sequences=num_return_sequences,
                          )         
    
    # Get system prompt text
    with open(system_prompt_loc, 'r') as file:
        system_prompt = file.read()

    # Load either raw post excel files, or intermediate results
    if not raw_csv_or_intermediate:
        user_input = input("Load raw csv files (c) or intermediate results (i)?")
    else:
        user_input = raw_csv_or_intermediate
    if user_input == 'c':
        df = pd.read_csv(input_path)
        df['frames'] = None
    elif user_input == 'i':
        filename = intermediate_path
        df = pd.read_csv(filename)
    else:
        raise ValueError("Expected 'c' or 'i' as input.")

    print('Data loaded.')

    # Post content is in 'post_col' column
    labeled_df = pd.read_csv(labeled_data_path)

    # Loop through chunks of the df, getting and saving results for each chunk.
    # The only reason this is done chunkwise is to prevent losing all the results if an error occurs

    # Define chunk size, for example, 5000 rows per chunk
    chunk_size = chunk_size

    # Create a DataFrame of unique messages
    # Replace URLs with a placeholder
    # import pdb; pdb.set_trace()
    # Drop nas first
    df = df[df[post_col].notna()]
    df['normalized_text'] = df[post_col].apply(lambda x: re.sub(r'http\S+', 'http:URL', x))
    # Replace Twitter usernames with a placeholder
    df['normalized_text'] = df['normalized_text'].apply(lambda x: re.sub(r'@\S+', '@USER', x))
    # Deduplicate based on the normalized text
    unique_df = df.loc[pd.isnull(df['frames'])].drop_duplicates(subset='normalized_text').reset_index()
    unique_df = unique_df.sample(frac=1).reset_index(drop=True)  # shuffle the unique_df, so we gather theories in random order

    # Update chunk_size if given sufficiently small data.
    if unique_df.shape[0] <= chunk_size:
        chunk_size = unique_df.shape[0] // 2

    # Create chunks for the unique DataFrame
    chunks = [x for x in range(0, unique_df.shape[0], chunk_size)]

    # Initialize a results dictionary to keep track of unique results
    results_dict = {}

    # Initialize a single progress bar for the entire data processing
    total_chunks = len(chunks) - 1
    pbar = tqdm(total=total_chunks, desc="Overall progress")

    # Process each chunk of unique messages
    for i in range(len(chunks)-1):
        # Apply the function to each unique message in the chunk and store the result in the results_dict
        results_dict_chunk = unique_df.iloc[chunks[i]:chunks[i+1]].apply(lambda row: {row['normalized_text']: get_results_from_row(row, 'normalized_text', model, labeled_df, system_prompt)}, axis=1).values

        # Concatenate the result dictionaries of this chunk into the main results_dict
        for res in results_dict_chunk:
            results_dict.update(res)

        # Map the results_dict to the 'Message' column in the full df
        mask = df['frames'].isnull()
        df.loc[mask, 'frames'] = df.loc[mask, 'normalized_text'].map(results_dict)

        # Save intermediate results to a csv file
        df.to_csv(intermediate_path, index=False, quoting=csv.QUOTE_ALL)

        # Update the progress bar after each chunk is processed
        pbar.update(1)
   
    # import pdb; pdb.set_trace()
    # If there are any rows left over, process them
    results_dict_chunk = unique_df.iloc[chunks[-1]:].apply(lambda row: {row['normalized_text']: get_results_from_row(row, 'normalized_text', model, labeled_df, system_prompt)}, axis=1).values

    # Close the progress bar
    pbar.close()

    for res in results_dict_chunk:
        results_dict.update(res)

    mask = df['frames'].isnull()
    df.loc[mask, 'frames'] = df.loc[mask, 'normalized_text'].map(results_dict)

    # Save final results
    # Drop the 'normalized_text' column since not needed anymore
    # pdb.set_trace()
    df.drop(columns=['normalized_text'], inplace=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    return df

if __name__ == "__main__":
    args = parse_args()
    process_posts(args.input_path, args.post_col, args.intermediate_path, args.output, args.labeled_data_path)
