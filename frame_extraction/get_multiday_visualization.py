# Imports
from visualize_frame_clusters_across_time import visualize_frame_cluster_across_time
import os

# Config: Time series visualization settings
topic = 'EconomicFreedomFighters'
last_day = '2024-03-25'
num_bins = 12
time_col = 'CreatedTime'
round_to_nearest = 'H'
output_dir = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections', 'frame_extraction_analysis', 'outputs')
original_data_loc = os.path.join('/', 'zfs', 'disinfo', 'Monitoring', 'Africa_Elections')
api_key_loc = os.path.join('/', 'home', 'cehrett', '.apikeys', 'openai_api_key.txt')
id_col = 'UniversalMessageId'
username='cehrett'
query_theories = ['There is corruption in the African National Congress', 
                  'The Zondo Commission findings are a threat to the ANC', 
                  'The Judicial Commission of Inquiry into Allegations of State Capture must be taken seriously']

# Get single-day visualization figures:
fig1_html, fig2_html, fig3_html = visualize_frame_cluster_across_time(
    frame_cluster_results_loc=os.path.join(output_dir, topic, last_day, 'frame_cluster_results_across_days.csv'),
    original_data_loc=os.path.join(original_data_loc, topic, last_day + '.csv'),
    frame_cluster_embeddings_loc=os.path.join(output_dir, topic, last_day, 'frame_embeddings.json'),
    num_bins=num_bins,
    round_to_nearest=round_to_nearest,
    time_col=time_col,
    id_col=id_col,
    num_fcs_to_display=8,
    figures_output_loc=None,
    username=username,
    api_key_loc=api_key_loc,
    query_theories=query_theories,
    multiday=False,
    return_figures=True
    )


# Get multi-day visualization figures:
fig4_html, fig5_html, fig6_html = visualize_frame_cluster_across_time(
    frame_cluster_results_loc=output_dir,
    original_data_loc=original_data_loc,
    frame_cluster_embeddings_loc=output_dir,
    num_bins=num_bins * 3,
    round_to_nearest=round_to_nearest,
    time_col=time_col,
    id_col=id_col,
    num_fcs_to_display=8,
    figures_output_loc=None,
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
with open(os.path.join(output_dir, topic, last_day, 'frame_cluster_activity_across_time.html'), 'w') as f:
    f.write(html_template)

import pdb; pdb.set_trace() 


