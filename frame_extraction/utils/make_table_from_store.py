# Imports
import pandas as pd
import os
import argparse

# Parse arguments
def parse_args():
    """
    Parses command-line arguments.

    Returns:
    - argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get cluster descriptions for frame clusters.")
    parser.add_argument("--clusters_input", type=str, required=True, help="The input file containing the frame clusters."
                        "Should be a csv with columns 'frames', 'cluster_labels', and 'description'.")
    parser.add_argument("--output_file", type=str, required=True, help="The output html filepath to save the results.")
    parser.add_argument("--n_samp", type=int, default=6, help="The number of unique texts to sample for each cluster.")
    
    return parser.parse_args()


def populate_posts(df, topic_dir, id_col='UniversalMessageId', text_col='Message', chunk_size=10000):
    """
    Populates the 'post' column in the dataframe with text from csv files in the topic_dir.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing an 'id' column.
    topic_dir (str): Path to the directory containing csv files.
    id_col (str): Name of the column in the csv files that matches the 'id' column in the dataframe.
    text_col (str): Name of the column in the csv files that contains the text to be added to the dataframe.
    chunk_size (int): Size of the chunks to read from the csv files to avoid memory issues.
    
    Returns:
    pandas.DataFrame: Updated DataFrame with a 'post' column containing the matched text.
    """
    # Initialize the 'post' column with None
    df['post'] = None

    # Get df length, to verify that it isn't changed during the process
    df_len = len(df)

    # Get a set of ids that need to be matched
    ids_to_match = set(df['id'])

    # Iterate through each CSV file in the directory
    for filename in os.listdir(topic_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(topic_dir, filename)

            # Read CSV file in chunks to avoid memory issues
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Filter chunk to include only rows with ids we are looking for
                matched_chunk = chunk[chunk[id_col].isin(ids_to_match)]

                # Merge matched_chunk with the dataframe to get the text values; make sure merged_df keeps df's index
                merged_df = df.merge(matched_chunk[[id_col, text_col]], left_on='id', right_on=id_col, how='left')

                # Preserve the original index
                merged_df.set_index(df.index, inplace=True)

                # Update the 'post' column in the original dataframe
                df.loc[merged_df['post'].isnull() & merged_df[text_col].notnull(), 'post'] = merged_df[text_col]

                # Remove matched ids from the set of ids to match
                ids_to_match -= set(matched_chunk[id_col])

                # Break the loop if all ids have been matched
                if not ids_to_match:
                    break
        if not ids_to_match:
            break

    # Check if all ids have been matched
    if ids_to_match:
        print(f"Warning: {len(ids_to_match)} ids were not matched.")

    # Check if the length of the dataframe has changed; if so, throw an error
    if len(df) != df_len:
        raise ValueError("The length of the dataframe has changed unexpectedly during the process. Please check the code.")

    return df


def make_table(clusters_input, output_file, n_samp, topic_dir, id_col='UniversalMessageID', text_col='Message'):

    # Read the input file
    df = pd.read_csv(clusters_input)

    # Keep only up to n_samp unique rows per `cluster_labels` value
    df = df.groupby('cluster_labels').head(n_samp)

    # Add a `post` column with `populate_posts` function
    df = populate_posts(df, topic_dir, id_col=id_col, text_col=text_col, chunk_size=10000)

    # Aggregate the `frames` and `post` columns by `cluster_labels`, so that there is one row per `cluster_labels` value
    df = df.groupby('cluster_labels').agg({'frames': lambda x: '<hr>'.join(x), 'post': lambda x: '<hr>'.join(x)}).reset_index()

    # Drop all columns except 'cluster_labels', `post` and 'frames'
    df = df[['cluster_labels', 'post', 'frames']]

    # Create an html file that displays the resulting df
    df.to_html(output_file, index=False, escape=False)
               
    # Print the location of the output log file
    print(f"HTML output log file saved to {output_file}.")


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    make_table(args.clusters_input, args.output_file, args.n_samp, args.topic_dir, args.id_col, args.text_col)