# Imports
import pandas as pd
import numpy as np
import argparse
from openai import OpenAI
import json
from IPython.display import display
from IPython.display import Markdown
import markdown
from .utils.load_llm_model import prepare_to_load_model
from .utils.token_utils import num_tokens_from_messages
from .utils.token_utils import partition_prompt


# Define system prompt
get_cluster_description_prompt = """\
# CONTEXT
You are a data scientist working for a research organization that studies disinformation campaigns. \
Your team is analyzing a set of "frames" found in social media posts. \
A "frame" is a factual or moral claim of broad social significance. \
The user will provide a table representing clusters of "frames" \
expressed by a corpus of social media posts.
The left column of the table is the cluster label id. \
The right column of the table is a sample of up to {n_samp} unique frames \
from that cluster. The frames are separated by "<br>".

# OBJECTIVE
You must produce a single frame for each cluster in the table. \
The frame you produce should capture the overall meaning of the frames in that cluster. \
For each cluster in the table, you will produce a frame that is semantically equivalent to the frames in that cluster. \
Frames always should be a complete sentence with a subject, verb, and object. \

# RESULT
You must respond in JSON format. Return only JSON, with no additional text. \
Your response should be a dictionary with a key for each cluster id. \
The value for each key should be the frame (a string) that you produce to describe that cluster. \
"""


def create_markdown_table(df, n_samp=5, include_descriptions=False):
    """
    Creates individual Markdown table for a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame.
    - n_samp (int): The number of unique texts to sample for each cluster.
    - include_descriptions (bool): Whether to include descriptions in the table.

    Returns:
    - markdown_table (str): A Markdown table string.
    """

    # Initialize the Markdown table for the current DataFrame
    markdown_table = [f"## Table", "| Cluster | Frames |", "| --- | --- |"]
    if include_descriptions:
        markdown_table[1] += " Description |"
        markdown_table[2] += " --- |"

    # Drop rows with missing 'frames' values
    df.dropna(subset=['frames'], inplace=True)

    # Drop rows with cluster label -1
    df = df[df['cluster_labels'] != -1]
    
    # Group by 'cluster_labels' and select up to `n_samp` unique 'frame' texts for each category
    for cluster_label, group in df.groupby('cluster_labels'):
        # Sample up to `n_samp` unique texts, handling cases with fewer than `n_samp` texts available
        sampled_texts = np.random.choice(group['frames'].unique(), size=min(n_samp, len(group['frames'].unique())), replace=False)
        # Get the cluster description
        cluster_description = group['description'].iloc[0] if include_descriptions else ''
        # Format the row for this cluster_label
        row = f"| {cluster_label} | {'<br>'.join(sampled_texts)} |"
        if include_descriptions:
            row += f" {cluster_description} |"
        markdown_table.append(row)

    markdown_table = '\n'.join(markdown_table) + '\n'
    
    return markdown_table


def get_llm_descriptions(markdown_table, 
                         system_prompt, 
                         model='gpt-4-turbo-preview'):
    """
    Writes cluster descriptions using an OpenAI model.

    Args:
        markdown_table (stf): A markdown table.
        system_prompt (str): The system prompt for producing the descriptions.
        model (str): The name of the GPT model to use.

    Returns:
        str: The descriptions produced by the GPT model.
    """

    # Check token length of prompt, and issues sub-processes if necessary
    message = [
        {"role": "system", "content": get_cluster_description_prompt},
        {"role": "user", "content": f"# TABLES\n{markdown_table}"}
    ]

    tokens = num_tokens_from_messages(message, model)

    responses = []
    
    if tokens >= 63000:
        print("Message length sufficiently large, creating sub-processes")
        partitioned_markdown_tables = partition_prompt(message, model)
    
        # For each partitioned markdown table, make separate API call
        for markdown_table in partitioned_markdown_tables:
            client = OpenAI()

            completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"# TABLES\n{markdown_table}"}
            ]
            )
            
            # Append each API call message content to response string (removing the ```json and ending ```)
            raw_string = completion.choices[0].message.content.replace('```json\n', '')
            processed_string = raw_string.replace('```', '')
            responses.append(processed_string)

        # String cleaning and processing
        responses = [s for s in responses if s.strip() and s != '{}']

        responses = [eval(dictionary) for dictionary in responses if eval(dictionary)]

        for dictionary in responses:
            responses[0].update(dictionary)

        response = '```json\n' + json.dumps(responses[0]) + '\n```'


    else:
        client = OpenAI()
                
        completion = client.chat.completions.create(
        model=model,
        messages=message
        )

        response += completion.choices[0].message.content

    print(f'Descriptions produced by {model}:')
    print(f'{response}')

    return response


def convert_string_to_dict(input_str):
    """
    Converts a string containing JSON representation
    of a dictionary into a Python dictionary.
    
    Parameters:
    - input_str (str): The input string containing the JSON data.
    
    Returns:
    - dict: A dictionary.
    """
    # Extract the JSON part of the input string
    try:
        json_str = input_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    except IndexError:
        # If the input string appears to be bordered by square brackets, extract the JSON string
        if "[" in input_str[:50] and "]" in input_str[-50:]:
            # Find the outermost square brackets and extract the JSON string
            start_idx = input_str.find("[")
            end_idx = input_str.rfind("]")
            json_str = input_str[start_idx:end_idx+1]
        elif input_str.startswith("{"):
            json_str = input_str
            if not input_str.endswith("}"):
                json_str += "\"}"
    
    # Parse the JSON string into a Python object
    data = json.loads(json_str)

    # Convert keys to int
    data = {int(k): v for k, v in data.items()}
    
    return data


def add_cluster_descriptions_to_df(df, descriptions):
    """
    Adds cluster descriptions to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the clusters.
        descriptions (dict): A dictionary mapping cluster labels to descriptions.

    Returns:
        pd.DataFrame: The DataFrame with descriptions added.
    """
    # Add a new column 'description' to the DataFrame
    df['description'] = df['cluster_labels'].map(descriptions)
    
    return df


def display_markdown_cluster_descriptions(df, n_samp=5):
    """
    Displays cluster descriptions in Markdown format along with n_samp unique examples of frames from each cluster.
    Requires that cluster descriptions have already been gathered. I.e., the input df should already have a 'description' column,
    in addition to the 'cluster_labels' and 'frames' columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the clusters.
        n_samp (int): The number of unique texts to sample for each cluster.
    """
    # Create a Markdown table
    markdown_table = create_markdown_table(df, n_samp=n_samp, include_descriptions=True)
    
    # Display the Markdown table
    display(Markdown(markdown_table))

    return markdown_table


def create_html_output_log(markdown_table, model, output_loc='cluster_descriptions.html'):
    """
    Creates an HTML output log file containing the input Markdown tables and the final Markdown table, converted to HTML.
    """
    # Convert Markdown tables to HTML
    html_table = markdown.markdown(markdown_table, extensions=['tables']) 

    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    table, th, td {{
      border: 1px solid black;
    }}
    th, td {{
      padding: 10px;
      text-align: left;
    }}
    </style>
    </head>
    <body>
    
    <h2>Table of descriptions of frame-clusters, produced by {model}</h2>
    {html_table}
    
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(output_loc, "w") as file:
        file.write(html_content)

    # Print the location of the output log file
    print(f"HTML output log file saved to {output_loc}.")


def parse_args():
    """
    Parses command-line arguments.

    Returns:
    - argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get cluster descriptions for frame clusters.")
    parser.add_argument("--input_file", type=str, required=True, help="The input file containing the frame clusters.")
    parser.add_argument("--output_file", type=str, required=True, help="The output file to save the results.")
    parser.add_argument("--n_samp", type=int, default=10, help="The number of unique texts to sample for each cluster.")
    parser.add_argument("--api_key_loc", type=str, default='./openai_api_key.txt', help="The location of the OpenAI API key.")
    parser.add_argument("--model", type=str, default='gpt-4-turbo-preview', help="The name of the GPT model to use.")
    
    return parser.parse_args()


def get_cluster_descriptions(input_file, output_file, api_key_loc, n_samp, model):

    # Load the OpenAI API key
    prepare_to_load_model(api_key_loc=api_key_loc)
    
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Create a Markdown table
    markdown_table = create_markdown_table(df, n_samp=n_samp)

    descriptions = get_llm_descriptions(markdown_table, get_cluster_description_prompt, model=model)
    
    # Convert the descriptions to a dictionary
    descriptions_dict = convert_string_to_dict(descriptions)
    
    # Add the cluster descriptions to the DataFrame
    df = add_cluster_descriptions_to_df(df, descriptions_dict)
    
    # Save the DataFrame with descriptions to the output file
    df.to_csv(output_file, index=False)
    
    print(f"Cluster descriptions saved to {output_file}.")

    # Display the cluster descriptions in Markdown format
    markdown_table = display_markdown_cluster_descriptions(df, n_samp=n_samp)

    # Create an HTML output log file
    create_html_output_log(markdown_table, model)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    get_cluster_descriptions(input_file=args.input_file, 
                             output_file=args.output_file, 
                             api_key_loc=args.api_key_loc, 
                             n_samp=args.n_samp, 
                             model=args.model)