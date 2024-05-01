# Imports
import os
import argparse
from .visualize_frame_clusters_across_time import visualize_frame_cluster_across_time

def get_single_and_multiday_visualizations(frame_cluster_results_loc, 
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
                                           query_theories, 
                                           topic=None, 
                                           last_day=None):
    
    # Get single-day visualization figures:
    fig1_html, fig2_html, fig3_html = visualize_frame_cluster_across_time(
        frame_cluster_results_loc=os.path.join(frame_cluster_results_loc, topic, last_day, 'frame_cluster_results_across_days.csv'),
        original_data_loc=original_data_loc,
        frame_cluster_embeddings_loc=frame_cluster_embeddings_loc,
        num_bins=num_bins,
        round_to_nearest=round_to_nearest,
        time_col=time_col,
        id_col=id_col,
        num_fcs_to_display=num_fcs_to_display,
        figures_output_loc=figures_output_loc,
        username=username,
        api_key_loc=api_key_loc,
        query_theories=query_theories,
        multiday=False,
        return_figures=True,
        last_day=last_day,
        topic=topic
    )

    # Get multi-day visualization figures:
    fig4_html, fig5_html, fig6_html = visualize_frame_cluster_across_time(
        frame_cluster_results_loc=frame_cluster_results_loc,
        original_data_loc=frame_cluster_results_loc,
        frame_cluster_embeddings_loc=frame_cluster_embeddings_loc,
        num_bins=num_bins * 3,
        round_to_nearest=round_to_nearest,
        time_col=time_col,
        id_col=id_col,
        num_fcs_to_display=num_fcs_to_display,
        figures_output_loc=figures_output_loc,
        username=username,
        api_key_loc=api_key_loc,
        query_theories=query_theories,
        multiday=True,
        return_figures=True,
        topic=topic,
        last_day=last_day
    )

    # Write figures html into a file
    # First define template
    html_template = f"""
    <html>
    <head>
    <title>Combined Plotly Figures</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    <h1>Frame cluster activity over time</h1>
    <div>Number of Time Bins: {num_bins}</div>
    <div>Queries Used for Figures 3 and 6:</div>
    <ul>
    {''.join(f'<li>{query}</li>' for query in query_theories) if query_theories is not None else 'None provided'}
    </ul>
    {fig1_html}
    {fig2_html}
    {fig3_html}
    {fig4_html}
    {fig5_html}
    {fig6_html}
    </body>
    </html>
    """

    # Write html to file
    with open(os.path.join(figures_output_loc, 'frame_cluster_activity_across_time.html'), 'w') as f:
        f.write(html_template)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize frame clusters across time with various settings.")
    parser.add_argument('--topic', required=True, help='Topic for the analysis. Should match the directory name in the output directory')
    parser.add_argument('--last_day', required=True, help='Last day for the data to be analyzed. YYYY-MM-DD format')
    parser.add_argument('--num_bins', type=int, required=True, help='Number of time bins per day')
    parser.add_argument('--time_col', required=True, help='Column name for the time data')
    parser.add_argument('--round_to_nearest', required=True, help='Rounding granularity for time data')
    parser.add_argument('--id_col', required=True, help='Column name for the ID')
    parser.add_argument('--username', required=True, help='Username for access control')
    parser.add_argument('--api_key_loc', required=True, help='Location of the API key')
    parser.add_argument('--num_fcs_to_display', type=int, default=8, help='Number of frame clusters to display')
    parser.add_argument('--query_theories', required=True, help='Semicolon-separated list of query theories')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--original_data_loc', required=True, help='Location of the original data')

    # Config: Time series visualization settings
    # topic = 'EconomicFreedomFighters'
    # last_day = '2024-03-25'
    # num_bins = 12
    # time_col = 'CreatedTime'
    # round_to_nearest = 'H'
    # output_dir = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections', 'frame_extraction_analysis', 'outputs')
    # original_data_loc = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections')
    # api_key_loc = os.path.join('/', 'home', 'cehrett', '.apikeys', 'openai_api_key.txt')
    # id_col = 'UniversalMessageId'
    # username='cehrett'
    # query_theories = ['There is corruption in the African National Congress', 
    #                 'The Zondo Commission findings are a threat to the ANC', 
    #                 'The Judicial Commission of Inquiry into Allegations of State Capture must be taken seriously']

    args = parser.parse_args()
    print(f"Query theories: {args.query_theories.split(';')}")
    input('Press Enter to continue if the query theories are correct.')
    get_single_and_multiday_visualizations(frame_cluster_results_loc=os.path.join(args.output_dir, args.topic, args.last_day, 'frame_cluster_results.csv'),
                                           original_data_loc=args.original_data_loc,
                                           frame_cluster_embeddings_loc=os.path.join(args.output_dir, args.topic, args.last_day, 'frame_cluster_embeddings.csv'),
                                           num_bins=args.num_bins,
                                           round_to_nearest=args.round_to_nearest,
                                           time_col=args.time_col,
                                           id_col=args.id_col,
                                           num_fcs_to_display=args.num_fcs_to_display,
                                           figures_output_loc=os.path.join(args.output_dir, args.topic, args.last_day),
                                           username=args.username,
                                           api_key_loc=args.api_key_loc,
                                           query_theories=args.query_theories.split(';'),
                                           topic=args.topic,
                                           last_day=args.last_day
                                          )
