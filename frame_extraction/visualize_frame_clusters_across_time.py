import pandas as pd
import os
import argparse
from .utils import clustering_tools as ct
import plotly.io as pio
from dateutil import parser


def save_figures_to_html(query_theories, num_bins, figures_output_loc, fig1_html, fig2_html, fig3_html=None):
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


def generalized_date_parser(date):
    try:
        # Attempt to parse the date string using the dateutil parser
        return pd.to_datetime(parser.parse(str(date)), utc=True)
    except Exception:
        # If parsing fails, handle the failure case here
        return pd.NaT


def visualize_frame_cluster_across_time(frame_cluster_results_loc, 
                                        original_data_loc, 
                                        frame_cluster_embeddings_loc, 
                                        num_bins, 
                                        round_to_nearest, 
                                        time_col, 
                                        id_col, 
                                        num_fcs_to_display, 
                                        username, 
                                        api_key_loc, 
                                        last_day=None,
                                        query_theories=None,
                                        figures_output_loc=None, 
                                        min_time=None,
                                        max_time=None,
                                        multiday=False,
                                        topic=None,
                                        return_figures=False,
                                        bin_times=True
                                       ):
    
    """
    Visualizes frame clusters across time.
    
    Parameters:
    frame_cluster_results_loc (str): Location of the frame cluster results CSV file.
    original_data_loc (str): Location of the original data CSV file. Should have time column in a format that can be fed to pd.to_datetime().
    frame_cluster_embeddings_loc (str): Location of the frame embeddings JSON file. Needed if and only if queries are provided, to see frame-clusters relevant to those queries.
    num_bins (int): Number of time bins into which to divide the data.
    round_to_nearest (str): Round the time bins (used for plot labels).
    time_col (str): Name of the time column in the data.
    id_col (str): Name of the ID column in the data.
    num_fcs_to_display (int): Number of frame clusters to display.
    username (str): Username for API access. Needed only if queries are provided.
    api_key_loc (str): Location of the API key. Needed only if queries are provided.
    last_day (str): The last day of analysis, in the format 'YYYY-MM-DD'. Required if multiday is set.
    query_theories (list): List of theories to query. If not provided, the script will skip steps related to query processing.
    figures_output_loc (str): Location to save the output HTML file with figures.
    min_time (str): Minimum time to consider for binning. If None, the minimum time in the data will be used.
    max_time (str): Maximum time to consider for binning. If None, the maximum time in the data will be used.
    multiday (bool): If this flag is set, the script will look for frame cluster results and original data in a multiday format.
    topic (str): The topic of analysis. Required if multiday is set.
    return_figures (bool): If True, the function will return the figures' html. Otherwise, it will save them to an HTML file.
    bin_times (bool): If True, the function will bin the times into num_bins. If False, the function will not bin the times.
    """
    
    if last_day:
        last_day = pd.to_datetime(last_day)

    # Get frame clusters loaded
    if multiday==False:
        fc_df = pd.read_csv(frame_cluster_results_loc)
        if 'embeddings' in fc_df.columns:
            fc_df.drop(['embeddings'], axis=1, inplace=True)
    else:
        # if analysis is multiday, then frame_cluster_results_loc is a directory
        # containing a subdirectory for each topic, which in turn have a subdirectory
        # for day of analysis, with day subdirectory names in the format 'YYYY-MM-DD'.
        # The last_day argument is the last day of analysis.
        # The topic argument is the topic of analysis.
        # We will load the frame cluster results for the last day of analysis, as 
        # well as for up to seven days prior to that, if available.
        

        # Get the topic of analysis
        if topic is None:
            raise ValueError('If multiday analysis is requested, the topic argument must be provided.')
        
        # Get the frame cluster results for the last day of analysis
        last_day_dir = os.path.join(frame_cluster_results_loc, topic, last_day.strftime('%Y-%m-%d'))
        if not os.path.exists(last_day_dir):
            raise ValueError(f'The directory {last_day_dir} does not exist.')
        fc_df = pd.read_csv(os.path.join(last_day_dir, 'frame_cluster_results.csv'))
        if 'embeddings' in fc_df.columns:
            fc_df.drop(['embeddings'], axis=1, inplace=True)
        fc_df['day'] = last_day.strftime('%Y-%m-%d')

        # Get the frame cluster results for up to seven days prior to the last day of analysis
        for i in range(1, 8):
            day = last_day - pd.DateOffset(days=i)
            day_dir = os.path.join(frame_cluster_results_loc, topic, day.strftime('%Y-%m-%d'))
            if os.path.exists(day_dir):
                day_fc_df = pd.read_csv(os.path.join(day_dir, 'frame_cluster_results.csv'))
                if 'embeddings' in day_fc_df.columns:
                    day_fc_df.drop(['embeddings'], axis=1, inplace=True)
                day_fc_df['day'] = day.strftime('%Y-%m-%d')
                fc_df = pd.concat([fc_df, day_fc_df], ignore_index=True)
            else:
                print(f'The directory {day_dir} does not exist. Skipping.')

    # Get original data (with dates) loaded
    if multiday==False:
        og_df = pd.read_csv(original_data_loc)
    else:
        og_df = pd.read_csv(os.path.join(original_data_loc, topic, last_day.strftime('%Y-%m-%d'), 'frame_extraction_results.csv'))

        # Load the original data for up to the last seven days
        for i in range(1, 8):
            day = last_day - pd.DateOffset(days=i)
            day_dir = os.path.join(original_data_loc, topic, day.strftime('%Y-%m-%d'), 'frame_extraction_results.csv')
            if os.path.isfile(day_dir):
                day_data = pd.read_csv(day_dir)
                og_df = pd.concat([og_df, day_data], ignore_index=True)

    # Drop duplicate rows
    og_df.drop_duplicates(inplace=True)

    # Convert time col to datetime
    try:
        og_df[time_col] = og_df[time_col].apply(generalized_date_parser)
    except Exception as e:
        print('ERROR: There was a problem converting the time into pandas datetime format.\n',\
              'Please ensure that the time column is formatted in such a way that pd.to_datetime() may be called on it.')
        raise e

    # Add the time col to the fc_df
    # Rename og_df id_col to match fc_df id_col
    og_df.rename(columns={id_col: 'id'}, inplace=True)
    fc_df = fc_df.merge(og_df[['id', time_col]], on='id', how='left')

    if bin_times:
        # Bin the times
        # Calculate the range and bin size
        if min_time == None:
            min_time = fc_df[time_col].min()
            print(f'Min time: {min_time}')
        if max_time == None:
            max_time = fc_df[time_col].max() + pd.to_timedelta(1, unit=round_to_nearest)
            print(f'Max time: {max_time}')
        range_seconds = (max_time - min_time).total_seconds()
        bin_size_seconds = range_seconds / (num_bins)
        print(f'Binning data into {num_bins} bins of size {bin_size_seconds} seconds each.')

        # Define a function to calculate the bin start time, rounded to the nearest 'round_to_nearest'
        def calculate_bin_start_time(datetime_value):
            bin_start_time = min_time + pd.to_timedelta(((datetime_value - min_time).total_seconds() // bin_size_seconds) * bin_size_seconds, unit='s')
            # Round to nearest 'round_to_nearest'
            bin_start_time = bin_start_time.round(round_to_nearest)
            return bin_start_time

        # Apply the function to create a new column with the bin start time
        fc_df['bin_start_time'] = fc_df[time_col].apply(calculate_bin_start_time)
        # Drop NaTs
        fc_df = fc_df[fc_df.bin_start_time.notna()]
    else:
        fc_df['bin_start_time'] = fc_df[time_col]
    
    # Get a new df that tells us what share of the conversation each theory occupies in each time period

    # Get unique values for cluster_labels and time_period
    unique_clusters = fc_df['cluster_labels'].unique()
    unique_periods = fc_df['bin_start_time'].unique()
    print(f'Unique periods: {unique_periods}')

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
    fig1 = ct.plot_theory_lines(ct.top_n_clusters(cluster_df, n=num_fcs_to_display), 
                                fc_df, 
                                theory_col='frames', 
                                add_sum_line=False, 
                                round_to_nearest=round_to_nearest,
                                plot_title='Frame clusters with greatest variation over time')

    # Now produce plot of fcs with most variation, limited to fcs that appear little in the first time block
    filter_dict = {min(cluster_df.columns): lambda x: x<0.01}
    fig2 = ct.plot_theory_lines(ct.top_n_clusters(cluster_df, 
                                                  n=num_fcs_to_display, 
                                                  filters=filter_dict), 
                                                  fc_df, 
                                                  round_to_nearest=round_to_nearest,
                                                  theory_col='frames', 
                                                  add_sum_line=False, 
                                                  plot_title='Frame clusters with greatest growth over time')
    
    # If there are queries for semantic search, use them to get another plot:
    if query_theories is not None:
        # Add embeddings to fc_df, so we can use semantic search to see fcs that are relevant to user-provided queries
        # if multiday is true, then get embeddings for each day of last seven days
        if multiday:
            embeddings_df = pd.read_json(os.path.join(frame_cluster_embeddings_loc, topic, last_day.strftime('%Y-%m-%d'), 'frame_embeddings.json'))
            for i in range(1, 8):
                day = last_day - pd.DateOffset(days=i)
                day_dir = os.path.join(frame_cluster_embeddings_loc, topic, day.strftime('%Y-%m-%d'), 'frame_embeddings.json')
                if os.path.isfile(day_dir):
                    day_embeddings = pd.read_json(day_dir)
                    shared_columns = embeddings_df.columns.intersection(day_embeddings.columns)
                    unique_columns_day_embeddings = day_embeddings.loc[:, ~day_embeddings.columns.isin(shared_columns)]
                    embeddings_df = pd.concat([embeddings_df, unique_columns_day_embeddings], axis=1)
                else:
                    print(f'The file {day_dir} does not exist. Skipping.')
        else:
            if topic and last_day:
                embeddings_df = pd.read_json(os.path.join(frame_cluster_embeddings_loc, topic, last_day.strftime('%Y-%m-%d'), 'frame_embeddings.json'))
            else:
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
                             round_to_nearest=round_to_nearest,
                             plot_title='Frame clusters most relevant to user-provided queries'
                                   )
        

    # Save plots to html file
    fig1_html = pio.to_html(fig1, full_html=False, include_plotlyjs=False)
    fig2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False)
    if query_theories is not None:
        fig3_html = pio.to_html(fig3, full_html=False, include_plotlyjs=False)
    else:
        fig3_html = None

    # If return_figures == True, return the figures' html. Otherwise, save them to an HTML file.
    if return_figures == True:
        return fig1_html, fig2_html, fig3_html
    else:
        save_figures_to_html(query_theories, num_bins, figures_output_loc, fig1_html, fig2_html, fig3_html)

        
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
    parser.add_argument("--multiday", action='store_true', help="If this flag is set, the script will look for frame cluster results and original data in a multiday format.")
    parser.add_argument("--last_day", default=None, help="The last day of analysis, in the format 'YYYY-MM-DD'. Required if multiday is set.")
    parser.add_argument("--topic", default=None, help="The topic of analysis. Required if multiday is set.")


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