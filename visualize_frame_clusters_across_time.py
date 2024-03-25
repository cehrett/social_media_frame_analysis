import pandas as pd
import os
import numpy as np
import argparse
import plotly.io as pio
import utils.clustering_tools as ct

def visualize_frame_cluster_across_time(frame_cluster_results_loc, 
                 original_data_loc, 
                 frame_cluster_embeddings_loc, 
                 num_bins, 
                 round_to_nearest, 
                 time_col, 
                 id_col, 
                 num_fcs_to_display, 
                 figures_output_loc, 
                 username, 
                 api_key_loc, 
                 query_theories):
    # Get frame clusters loaded
    fc_df = pd.read_csv(frame_cluster_results_loc)
    if 'embeddings' in fc_df.columns:
        fc_df.drop(['embeddings'], axis=1, inplace=True)

    # Get original data (with dates) loaded
    og_df = pd.read_csv(original_data_loc)

    # Drop duplicate rows
    og_df.drop_duplicates(inplace=True)

    # Convert time col to datetime
    try:
        og_df[time_col] = pd.to_datetime(og_df[time_col], format='mixed', utc=True)
    except Exception as e:
        print('ERROR: There was a problem converting the time into pandas datetime format.\n',\
              'Please ensure that the time column is formatted in such a way that pd.to_datetime() may be called on it.')
        raise e

    # Add the time col to the fc_df
    fc_df = fc_df.merge(og_df[[id_col, time_col]], on=id_col, how='left')
    # import pdb; pdb.set_trace()

    # Bin the times

    # Calculate the range and bin size
    min_time = fc_df[time_col].min()
    max_time = fc_df[time_col].max() + pd.to_timedelta(1, unit=round_to_nearest)
    range_seconds = (max_time - min_time).total_seconds()
    bin_size_seconds = range_seconds / (num_bins)

    # Define a function to calculate the bin start time, rounded to the nearest 'round_to_nearest'
    def calculate_bin_start_time(datetime_value):
        bin_start_time = min_time + pd.to_timedelta(((datetime_value - min_time).total_seconds() // bin_size_seconds) * bin_size_seconds, unit='s')
        # Round to nearest day
        bin_start_time = bin_start_time.round(round_to_nearest)
        return bin_start_time

    # Apply the function to create a new column with the bin start time
    fc_df['bin_start_time'] = fc_df[time_col].apply(calculate_bin_start_time)
    # Drop NaTs
    fc_df = fc_df[fc_df.bin_start_time.notna()]
    
    # Get a new df that tells us what share of the conversation each theory occupies in each time period

    # Get unique values for cluster_labels and time_period
    unique_clusters = fc_df['cluster_labels'].unique()
    unique_periods = fc_df['bin_start_time'].unique()

    # Initialize an empty dataframe with NaNs
    cluster_df = pd.DataFrame(index=unique_clusters, columns=unique_periods)

    # Calculate proportions
    for period in unique_periods:
        time_period_df = fc_df[fc_df['bin_start_time'] == period]
        total_rows_for_period = len(time_period_df)
        for cluster in unique_clusters:
            matching_rows = len(time_period_df[time_period_df['cluster_labels'] == cluster])
            proportion = matching_rows / total_rows_for_period
            cluster_df.at[cluster, period] = proportion

    # Produce plot of fcs with most variation over time        
    fig1 = ct.plot_theory_lines(ct.top_n_clusters(cluster_df, n=num_fcs_to_display), fc_df, theory_col='frames', add_sum_line=False, plot_title='Frame clusters with greatest variation over time')

    # Now produce plot of fcs with most variation, limited to fcs that appear little in the first time block
    filter_dict = {min(cluster_df.columns): lambda x: x<0.01}
    fig2 = ct.plot_theory_lines(ct.top_n_clusters(cluster_df, n=num_fcs_to_display, filters=filter_dict), fc_df, theory_col='frames', add_sum_line=False, plot_title='Frame clusters with greatest growth over time')
    
    # If there are queries for semantic search, use them to get another plot:
    if query_theories is not None:

        # Add embeddings to fc_df, so we can use semantic search to see fcs that are relevant to user-provided queries
        embeddings_df = pd.read_json(frame_cluster_embeddings_loc)
        # drop na columns of fc_df
        fc_df.dropna(subset='frames', inplace=True)
        fc_df['embeddings'] = fc_df['frames'].apply(lambda x: embeddings_df[x].tolist())
        del embeddings_df

        # Get embeddings of queries, use them to find relevant frame-clusters
        embeddings = ct.get_embeddings_of_query_theories(username, api_key_loc, query_theories, skip_verification=True)
        fig3 = ct.plot_theory_lines(ct.top_n_clusters(ct.compute_relevance(fc_df,embeddings,cluster_df), 
                                               n=num_fcs_to_display, toptype='relevance'), 
                             fc_df, 
                             theory_col='frames',
                             add_sum_line=True,
                             plot_title='Frame clusters most relevant to user-provided queries'
                                   )

    # Save plots to html file
    fig1_html = pio.to_html(fig1, full_html=False, include_plotlyjs=False)
    fig2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
    if query_theories is not None:
        fig3_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
    else:
        fig3_html = None

    html_template = f"""
    <html>
    <head>
    <title>Combined Plotly Figures</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    <h1>Frame cluster activity over time</h1>
    <div>Number of Time Bins: {num_bins}</div>
    <div>Queries Used for Figure 3:</div>
    <ul>
    {''.join(f'<li>{query}</li>' for query in query_theories) if query_theories is not None else 'None provided'}
    </ul>
    {fig1_html}
    {fig2_html}
    {fig3_html}
    </body>
    </html>
    """

    # Save Combined HTML to File
    with open(figures_output_loc, "w") as file:
        file.write(html_template)
        
        
def main():

    parser = argparse.ArgumentParser(description="Process and visualize frame cluster data.")
    parser.add_argument("--frame_cluster_results_loc", default=os.path.join('.', 'output', 'frame_cluster_results.csv'), help="Location of the frame cluster results CSV file.")
    parser.add_argument("--original_data_loc", default=os.path.join('.', 'data', 'sample_data_for_frame_extraction.csv'), help="Location of the original data CSV file. Should have time column in a format that can be fed to pd.to_datetime().")
    parser.add_argument("--frame_cluster_embeddings_loc", default=None, help="Location of the frame embeddings JSON file. Needed if and only if queries are provided, to see frame-clusters relevant to those queries.")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of time bins into which to divide the data.")
    parser.add_argument("--round_to_nearest", default='D', help="Round the time bins (used for plot labels).")
    parser.add_argument("--time_col", default='time', help="Name of the time column in the data.")
    parser.add_argument("--id_col", default='id', help="Name of the ID column in the data.")
    parser.add_argument("--num_fcs_to_display", type=int, default=5, help="Number of frame clusters to display.")
    parser.add_argument("--figures_output_loc", default=os.path.join('.', 'output', 'frame_clusters_over_time.html'), help="Location to save the output HTML file with figures.")
    parser.add_argument("--username", default=None, help="Username for API access. Needed only if queries are provided.")
    parser.add_argument("--api_key_loc", default=None, help="Location of the API key. Needed only if queries are provided.")
    parser.add_argument("--query_theories", nargs='+', default=None, help="List of theories to query. If not provided, the script will skip steps related to query processing.")


    args = parser.parse_args()

    visualize_frame_cluster_across_time(
        args.frame_cluster_results_loc,
        args.original_data_loc,
        args.frame_cluster_embeddings_loc,
        args.num_bins,
        args.round_to_nearest,
        args.time_col,
        args.id_col,
        args.num_fcs_to_display,
        args.figures_output_loc,
        args.username,
        args.api_key_loc,
        args.query_theories
    )
    
    
if __name__ == '__main__':
    main()