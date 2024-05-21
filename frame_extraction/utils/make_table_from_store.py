# Imports
import pandas as pd
import argparse

# Parse arguments
def parse_args():
    """
    Parses command-line arguments.

    Returns:
    - argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get cluster descriptions for frame clusters.")
    parser.add_argument("--input_file", type=str, required=True, help="The input file containing the frame clusters."
                        "Should be a csv with columns 'frames', 'cluster_labels', and 'description'.")
    parser.add_argument("--output_file", type=str, required=True, help="The output html filepath to save the results.")
    parser.add_argument("--n_samp", type=int, default=6, help="The number of unique texts to sample for each cluster.")
    
    return parser.parse_args()


def make_table(input_file, output_file, n_samp):

    # Read the input file
    df = pd.read_csv(input_file)

    # Drop duplicates with respect to the 'frames' and 'cluster_labels' columns
    df = df.drop_duplicates(subset=['frames', 'cluster_labels'])

    # Subsample to get a maximum of n_samp samples per cluster_labels value
    df = df.groupby('cluster_labels').apply(lambda x: x.sample(min(len(x), n_samp))).reset_index(drop=True)

    # Aggregate the 'frames' column by 'cluster_labels', so that there is one row per 'cluster_labels' value
    df = df.groupby('cluster_labels')['frames'].apply(lambda x: '<br>'.join(x)).reset_index()

    # Drop all columns except 'cluster_labels' and 'frames'
    df = df[['cluster_labels', 'frames']]

    # Create an html file that displays the resulting df
    df.to_html(output_file, index=False)
               
    # Print the location of the output log file
    print(f"HTML output log file saved to {output_file}.")


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    make_table(args.input_file, args.output_file, args.n_samp)