# Imports
import pandas as pd
import numpy as np
import os
import argparse
from openai import OpenAI
import json
import markdown


# Define system prompt
collapse_into_store_system_prompt = """\
# CONTEXT
You are a data scientist working for a research organization that studies disinformation campaigns. \
Your team is analyzing a text to identify clusters of "frames" that are semantically equivalent to a pre-existing store of frames. \
A "frame" is a factual or moral claim of broad social significance. \
The user will provide two tables that represent clusters of "frames", Table 0 - Store and Table 1 - Corpus. \
Each of the two tables represents a clustering of the frames expressed by social media posts in a text corpus.
The left column of each table is the cluster label. The middle column is a frame that summarizes the cluster. \
The right column is a sample of up to {n_samp} unique frames from that cluster. The sample frames are separated by "<br>".

# OBJECTIVE
You must find each cluster from Table 1 - Corpus which is semantically equivalent to some cluster in Table 0 - Store. \
Two clusters are semantically equivalent if they **mutually entail each other**, i.e., if their frames express the same meaning. \
For each cluster in Table 1 - Corpus, you must provide the corresponding cluster label from Table 0 - Store. \
If the cluster from Table 1 - Corpus is not semantically equivalent to any cluster in Table 0 - Store, \
then you most provide the cluster label "new_cluster" instead of an integer cluster label. \

# RESPONSE
You must respond in JSON format. Return only JSON, with no additional text. \
Your response should be a list of dictionaries. Each dictionary should have two key-value pairs: \
"table_0_cluster_label" and "table_1_cluster_label". \
For each cluster in Table 1 - Corpus, there should be a corresponding dictionary in your output. \
If you find that a cluster in Table 1 - Corpus is semantically equivalent to multiple clusters in Table 0 - Store, \
select the single cluster in Table 0 - Store that is most semantically equivalent to the cluster in Table 1. \
In other words, each cluster in Table 1 - Corpus should appear exactly once in your response. \
Clusters from Table 0 - Store are permitted to appear multiple times, if appropriate; \
i.e., it is possible that multiple clusters from Table 1 - Corpus are \
all semantically equivalent to the same cluster in Table 0 - Store. \
"""

collapse_across_dates_system_prompt = """\
# CONTEXT
You are a data scientist working for a research organization that studies disinformation campaigns. \
Your team is analyzing two text corpora to identify clusters of "frames" that are semantically equivalent. \
A "frame" is a factual or moral claim of broad social significance. \
The user will provide two tables that represent clusters of "frames", Table 0 and Table 1. \
Each of the two tables represents a clustering of the frames expressed by social media posts in one of the two text corpora.
The left column of each table is the cluster label. The right column of each table is a sample of up to {n_samp} unique frames \
from that cluster. The frames are separated by "<br>".

# OBJECTIVE
You must find each cluster from Table 1 which is semantically equivalent to some cluster in Table 0. \
Two clusters are semantically equivalent if they mutually entail each other, i.e., if their frames express the same meaning. \
For each cluster in Table 1 that is semantically equivalent to some cluster in the Table 0, \
you must provide the corresponding cluster label from Table 0.

# RESPONSE
You must respond in JSON format. Return only JSON, with no additional text. \
Your response should be a list of dictionaries. Each dictionary should have two key-value pairs: \
"table_0_cluster_label" and "table_1_cluster_label". \
For each cluster in Table 1 that is semantically equivalent to some cluster in Table 0, \
there should be a corresponding dictionary in your output. \
If you find that a cluster in Table 1 is semantically equivalent to multiple clusters in Table 0, \
select the single cluster in Table 0 that is most semantically equivalent to the cluster in Table 1. \
In other words, each cluster in Table 1 should appear at most once in your response. \
Clusters from Table 0 are permitted to appear multiple times, if appropriate; \
i.e., it is possible that multiple clusters from Table 1 are all semantically equivalent to the same cluster in Table 0. \
"""

collapse_single_day_system_prompt = """\
# CONTEXT
You are a data scientist working for a research organization that studies disinformation campaigns. \
Your team is analyzing a text corpus of social media posts to identify clusters of "frames" that are semantically equivalent. \
A "frame" is a factual or moral claim of broad social significance. \
The user will provide a tables that represents clusters of "frames". \
The table represents a clustering of the frames expressed by the social media posts in the text corpus.
The left column of the table is the cluster label. The right column of the table is a sample of up to {n_samp} unique frames \
from that cluster. The frames are separated by "<br>".

# OBJECTIVE
You must find each cluster from the table which is semantically equivalent to some other cluster in the table. \
Two clusters are semantically equivalent if they mutually entail each other, i.e., if their frames express the same meaning. \
For each cluster in the table that is semantically equivalent to some other cluster in the table, \
you must provide both cluster labels in your output. \

# RESPONSE
You must respond in JSON format. Return only JSON, with no additional text. \
Your response should be a list of dictionaries. Each dictionary should have two key-value pairs: \
"is_equivalent" and "equivalent_to". \
In all such semantically equivalent pairs, the higher of the two cluster labels should be listed first, as "is_equivalent". \
There should be one dictionary for each pair of semantically equivalent clusters in the table. \
If you find that a cluster is semantically equivalent to multiple other clusters in the table, \
select "equivalent_to" to be the single cluster in the table that is most semantically equivalent to the "is_equivalent" cluster. \
In other words, each cluster should be listed at most once in the "is_equivalent" position. \
Clusters are permitted to appear multiple times in the "equivalent_to" position, if appropriate. \
"""


# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Collapse semantically equivalent cluster labels within a day or across two days.')
    parser.add_argument('date', type=str, help='The date to be collapsed either internally or into the previous day\'s labels. YYYY-MM-DD format.')
    parser.add_argument('topic', type=str, help='Topic to collapse.')
    parser.add_argument('--across_days', action='store_true', 
                        help='Flag to indicate that the cluster labels should be collapsed into the previous day\'s. Cannot be true if a frame store is provided.')
    parser.add_argument('--store_loc', type=str, default=None, help='Location of the frame store CSV file. If provided, the cluster labels will be collapsed into the store.')
    parser.add_argument('--root_dir', type=str, default='outputs', help='Root directory containing the data.')
    parser.add_argument('--api_key_loc', type=str, default=os.path.expanduser('~/.apikeys/openai_api_key.txt'), help='Location of the API key file.')
    parser.add_argument('--model', type=str, default='gpt-4-turbo-preview', help='The OpenAI model to use for the chat completion.')

    return parser.parse_args()


def create_individual_markdown_table(df, n_samp=5, df_index='0'):
    """
    Creates individual Markdown table for a DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing a column 'cluster_labels', a column 'frames', and (optionally) a column 'description'.
    - n_samp (int): The number of unique texts to sample for each cluster.
    - df_index (str): The index of the DataFrame (used for MD table title)

    Returns:
    - markdown_table (str): A Markdown table string.
    """

    # Find whether the df has descriptions
    include_descriptions = 'description' in df.columns

    # Initialize the Markdown table for the current DataFrame
    if include_descriptions:
        markdown_table = [f"## Table {df_index}", "| Cluster | Description | Frames |", "| --- | --- | --- |"]
    else: 
        markdown_table = [f"## Table {df_index}", "| Cluster | Frames |", "| --- | --- |"]

    # Drop rows with missing 'frames' values
    df.dropna(subset=['frames'], inplace=True)

    # Drop rows with cluster label -1
    df = df[df['cluster_labels'] != -1]
    
    # Group by 'cluster_labels' and select up to `n_samp` unique 'frame' texts for each category
    for cluster_label, group in df.groupby('cluster_labels'):
        # If there are descriptions, get the cluster description (which is the same for all frames in the cluster)
        cluster_description = group['description'].iloc[0] if include_descriptions else ''
        # Sample up to `n_samp` unique texts, handling cases with fewer than `n_samp` texts available
        sampled_texts = np.random.choice(group['frames'].unique(), size=min(n_samp, len(group['frames'].unique())), replace=False)
        # Format the row for this cluster_label
        if include_descriptions:
            row = f"| {cluster_label} | {cluster_description} | {'<br>'.join(sampled_texts)} |"
        else:
            row = f"| {cluster_label} | {'<br>'.join(sampled_texts)} |"
        markdown_table.append(row)

    markdown_table = '\n'.join(markdown_table) + '\n'
    
    return markdown_table


def get_llm_clusters(markdown_tables, 
                     system_prompt, 
                     model='gpt-4-turbo-preview'):
    """
    Retrieves the cluster pairings using an OpenAI model.

    Args:
        markdown_tables (list): A list of markdown tables.
        collapse_across_dates_system_prompt (str): The system prompt for collapsing across dates.
        model (str): The name of the GPT model to use.

    Returns:
        str: The cluster pairings discovered by the GPT model.
    """
    
    client = OpenAI()

    # Get single string of all markdown tables
    newline_joined_markdown_tables = '\n'.join(markdown_tables)

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"# TABLES\n{newline_joined_markdown_tables}"}
    ]
    )

    print(f'Cluster pairings discovered by {model}:')
    print(f'{completion.choices[0].message.content}')
    return completion.choices[0].message.content


def convert_string_to_dict(input_str, labels=['is_equivalent', 'equivalent_to'], max_existing_label=None):
    """
    Converts a specially formatted string containing JSON representation
    of a list of dictionaries into a Python dictionary with `labels[0]`
    values as keys and `labels[1]` values as their corresponding values.
    
    Parameters:
    - input_str (str): The input string containing the JSON data.
    - labels (list of str): A list containing two strings, where the first
        string is the key for the dictionary keys and the second string
        is the key for the dictionary values.
    - max_existing_label (int): The maximum existing label in the previous day or frame store.
    
    Returns:
    - dict: A dictionary with `labels[0]` as keys and `labels[1]` as values.
    """
    # Extract the JSON part of the input string
    try:
        json_str = input_str.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    except IndexError:
        # Find the outermost square brackets and extract the JSON string
        start_idx = input_str.find("[")
        end_idx = input_str.rfind("]")
        json_str = input_str[start_idx:end_idx+1]
    
    # Parse the JSON string into a Python object
    data = json.loads(json_str)
    
    # Create the dictionary from the list of dictionaries. If item[labels[0]] is 'new_cluster', then assign a new label.
    result_dict = {}
    new_label = max_existing_label if max_existing_label else 0
    for item in data:
        if item[labels[0]] == 'new_cluster':
            new_label += 1
            result_dict[new_label] = int(item[labels[1]])
        else:
            result_dict[int(item[labels[0]])] = int(item[labels[1]])
    
    return result_dict


def rename_cluster_labels(df, mapping_dict):
    # Use the mapping_dict to rename the cluster labels in the df.
    # Since some cluster labels might get remapped to a label that itself needs to be remapped,
    # we will loop through the clusters labels from highest to lowest.

    # Loop through the cluster labels in the DataFrame
    for label in sorted(df['cluster_labels'].unique(), reverse=True):
        # If the label is in the mapping_dict keys, then rename it
        if label in mapping_dict:
            df['cluster_labels'] = df['cluster_labels'].replace(label, mapping_dict[label])

    return df


def rename_cluster_labels_across_days(dfs, mapping_dict):
    """
    Renames cluster labels in the second DataFrame (dfs[1]) using a mapping dictionary.
    The mapping dictionary should have cluster labels from the second DataFrame (dfs[1])
    as keys and corresponding cluster labels in the first DataFrame as values. I.e.,
    the mapping_dict should be a reversal of the dictionary created by the LLM model.

    Parameters:
    - dfs (list of pandas.DataFrame): A list containing two DataFrames. Each DataFrame
        should have at least a 'cluster_labels' column.
    - mapping_dict (dict): A dictionary with cluster labels from dfs[1] as keys and
        corresponding cluster labels in dfs[0] as values.

    Returns:
    - list of pandas.DataFrame: The modified list of DataFrames, with the second DataFrame
        having its cluster labels renamed according to the mapping_dict.
    """

    # Loop through the cluster labels in dfs[1]. If a label is -1 or a key in mapping_dict,
    # then continue. If a label is not yet a key in mapping_dict, then add it as a key,
    # with the value being a new label that is greater than the maximum value in the
    # cluster_labels column of dfs[0].
    max_label = dfs[0]['cluster_labels'].max()
    mapping_dict[-1] = -1
    for label in dfs[1]['cluster_labels'].unique():
        if label in mapping_dict:
            continue
        else:
            max_label += 1
            mapping_dict[label] = max_label

    # Apply the mapping_dict to the cluster_labels column of dfs[1]
    # Recall that mapping_dict has int keys and values, and the cluster_labels are ints
    dfs[1]['cluster_labels'] = dfs[1]['cluster_labels'].replace(mapping_dict)

    return dfs


def create_markdown_table_from_dict_and_dfs(mapping_dict, dfs, n_samp=5):
    """
    Creates a Markdown table string using a mapping dictionary and up to two DataFrames.
    The table will have four columns: The first column lists the cluster labels from
    the first (or only) DataFrame, the second column lists up to `n_samp` unique 'frames'
    values corresponding to each cluster label in that df. The third and fourth columns
    are similar but for the second (or only) DataFrame, using the mapping dictionary to
    match cluster labels.
    
    Parameters:
    - mapping_dict (dict): A dictionary with cluster labels from the first or only df as keys and
      corresponding cluster labels in the second or only df as values.
    - dfs (list of pandas.DataFrame): A list containing one or two DataFrames. Each DataFrame
      should have at least a 'cluster_labels' column and a 'frames' column.
    - n_samp (int): The number of unique 'frames' values to sample for each cluster.
    
    Returns:
    - str: A string representation of the constructed Markdown table.
    """
    # Check if all dataframes in dfs contain 'description' column
    include_descriptions = all(['description' in df.columns for df in dfs])

    # Start the Markdown table with headers
    if include_descriptions:
        markdown_str = "| Cluster A | Description A | Frames A | Cluster B | Description B | Frames B |\n"
        markdown_str += "|-------------|-----------------|------------|-------------|-----------------|------------|\n"
    else:
        markdown_str = "| Cluster A | Frames A | Cluster B | Frames B |\n"
        markdown_str += "|-------------|------------|-------------|------------|\n"
    
    # Loop through the keys of the mapping_dict
    for df1_label, df2_label in mapping_dict.items():
        # If there are descriptions, get the cluster descriptions (which are the same for all frames in the cluster) for each df
        if include_descriptions:
            df1_description = dfs[0][dfs[0]['cluster_labels'] == int(df1_label)]['description'].iloc[0] if int(df1_label) in dfs[0]['cluster_labels'].unique() else ''
            df2_description = dfs[-1][dfs[-1]['cluster_labels'] == int(df2_label)]['description'].iloc[0]

        # Get up to `n_samp` unique frames for each label in dfs[0]
        df1_frames = dfs[0][dfs[0]['cluster_labels'] == int(df1_label)]['frames'].unique()[:n_samp]
        # Get up to n_samp unique frames for the corresponding label in dfs[1]
        df2_frames = dfs[-1][dfs[-1]['cluster_labels'] == int(df2_label)]['frames'].unique()[:n_samp]
        
        # Join the frames with line breaks for Markdown display
        df1_frames_str = '<br>'.join(df1_frames)
        df2_frames_str = '<br>'.join(df2_frames)
        
        # Add the row for this pair of labels
        if include_descriptions:
            markdown_str += f"| {df1_label} | {df1_description} | {df1_frames_str} | {df2_label} | {df2_description} | {df2_frames_str} |\n"
        else:
            markdown_str += f"| {df1_label} | {df1_frames_str} | {df2_label} | {df2_frames_str} |\n"
    
    return markdown_str


def create_html_output_log(markdown_tables, markdown_final_table, model, output_loc='output_log.html'):
    """
    Creates an HTML output log file containing the input Markdown tables and the final Markdown table, converted to HTML.
    """
    # Convert Markdown tables to HTML
    html_tables = [markdown.markdown(table, extensions=['tables']) for table in markdown_tables]
    html_final_table = markdown.markdown(markdown_final_table, extensions=['tables'])

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
    
    <h2>Tables of individual day frame-clusters</h2>
    {''.join(html_tables)}
    
    <h2>Table of pairings chosen by {model}</h2>
    {html_final_table}
    
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(output_loc, "w") as file:
        file.write(html_content)

    # Print the location of the output log file
    print(f"HTML output log file saved to {output_loc}.")


def collapse(root_dir, topic, date_current, model, across_days=False, store_loc=None):

    # If across_days is True, then store_loc should not be provided
    if across_days and store_loc:
        raise ValueError("Cannot collapse cluster labels across days if a store_loc is provided.")

    # Load data
    # If a store_loc is provided or collapsing across days and a single-day collapse csv exists, then load the single-day-collapsed version.
    if (store_loc or across_days) and os.path.exists(os.path.join(root_dir, topic, date_current, 'frame_cluster_results_single_day.csv')):
        file_path_current = os.path.join(root_dir, topic, date_current, 'frame_cluster_results_single_day.csv')
    else:
        file_path_current = os.path.join(root_dir, topic, date_current, 'frame_cluster_results.csv')
    print(f"Loading data from {file_path_current}...")
    df_current = pd.read_csv(file_path_current)
    dfs = [df_current]

    # If collapsing across days, set the date_prev values and load the previous day's data; if using frame store, load that
    if across_days:
        date_prev = pd.to_datetime(date_current) - pd.Timedelta(days=1)
        date_prev = date_prev.strftime('%Y-%m-%d')
        file_path_prev = os.path.join(root_dir, topic, date_prev, 'frame_cluster_results_across_days.csv')
        df_prev = pd.read_csv(file_path_prev)
        dfs.insert(0, df_prev)
    elif store_loc:
        # Load the store DataFrame
        store_df = pd.read_csv(os.path.join(root_dir, topic, store_loc))
        dfs.insert(0, store_df)

    # Create individual Markdown tables for each DataFrame
    if across_days:
        markdown_tables = [create_individual_markdown_table(df, n_samp=5, df_index=str(i)) for i, df in enumerate(dfs)]
    elif store_loc:
        table_labels = ['0 - Store', '1 - Corpus']
        markdown_tables = [create_individual_markdown_table(df, n_samp=5, df_index=table_labels[i]) for i, df in enumerate(dfs)]
    else:
        markdown_table = create_individual_markdown_table(dfs[0], n_samp=5, df_index='')
        markdown_tables = [markdown_table]

    # Get cluster pairings from the OpenAI model
    # First, get the correct system prompt
    if across_days:
        system_prompt = collapse_across_dates_system_prompt
    elif store_loc:
        system_prompt = collapse_into_store_system_prompt
    else:
        system_prompt = collapse_single_day_system_prompt
    llm_output = get_llm_clusters(markdown_tables, system_prompt=system_prompt, model=model)

    # Get labels according to whether collapsing across days
    if across_days or store_loc:
        labels = ['table_0_cluster_label', 'table_1_cluster_label']
        max_existing_label = dfs[0]['cluster_labels'].max()
    else:
        labels = ['is_equivalent', 'equivalent_to']
        max_existing_label = None

    # Convert the LLM output string to a dictionary
    mapping_dict = convert_string_to_dict(llm_output, labels=labels, max_existing_label=max_existing_label)    

    # Remap the cluster labels according to the LLM output and save the final table
    if across_days or store_loc:
        # Reverse the mapping dictionary and rename cluster labels in the second DataFrame
        mapping_dict_rev = {v: k for k, v in mapping_dict.items()}
        markdown_final_table = create_markdown_table_from_dict_and_dfs(mapping_dict, dfs)
        if across_days:
            dfs = rename_cluster_labels_across_days(dfs, mapping_dict_rev)
        if store_loc:
            dfs[-1] = rename_cluster_labels(dfs[-1], mapping_dict_rev)
    else:
        markdown_final_table = create_markdown_table_from_dict_and_dfs(mapping_dict, dfs)
        dfs[-1] = rename_cluster_labels(dfs[-1], mapping_dict)

    # Save the modified second dataframe to a CSV file
    # Get suffix depending on whether collapsing across days or using store
    if across_days:
        suffix = '_across_days'
    elif store_loc:
        suffix = '_into_store'
    else:
        suffix = '_single_day'
    file_path_current_new = os.path.join(os.path.dirname(file_path_current), 'frame_cluster_results' + suffix + '.csv')
    dfs[-1].to_csv(file_path_current_new, index=False)
    print(f"Cluster labels in {file_path_current} have been collapsed and saved to {file_path_current_new}.")

    # If using a store, save the store DataFrame with the new cluster labels
    if store_loc:
        # First, back up the old store
        dfs[0].to_csv(os.path.join(root_dir, topic, store_loc[:-4] + '_backup.csv'), index=False)

        # Find which cluster labels are new (in dfs[-1] but not in dfs[0])
        new_labels = set(dfs[-1]['cluster_labels'].unique()) - set(dfs[0]['cluster_labels'].unique())
        # For each new label, randomly samply up to 30 rows from dfs[-1] that have unique frames and add them to dfs[0].
        for label in new_labels:
            # Get a df that is a subset of df[-1] with only the rows that have the new label
            new_label_df = dfs[-1][dfs[-1]['cluster_labels'] == label]
            # Deduplicate with respect to the frames column
            new_label_df = new_label_df.drop_duplicates(subset='frames')
            # Sample up to 30 rows
            new_label_df = new_label_df.sample(min(30, len(new_label_df)))
            # Add the new_label_df to dfs[0]
            dfs[0] = pd.concat([dfs[0], new_label_df], ignore_index=True)

        # Save the modified store DataFrame to a CSV file in the store_loc location
        dfs[0].to_csv(os.path.join(root_dir, topic, store_loc), index=False)
        print(f"Cluster labels in the store have been updated and saved to {store_loc}.")        

    # Create an HTML output log file
    if across_days:
        suffix = '_across_days'
    elif store_loc:
        suffix = '_into_store'
    else:
        suffix = '_single_day'
    output_loc = os.path.join(root_dir, topic, date_current, f'fc_collapse_log{suffix}.html')
    create_html_output_log(markdown_tables, markdown_final_table, model, output_loc=output_loc)

    print(f"Collapsing of cluster labels for {date_current} completed successfully.")


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Collapse the cluster labels
    collapse(args.root_dir, args.topic, args.date, args.model, across_days=args.across_days, store_loc=args.store_loc)


    

