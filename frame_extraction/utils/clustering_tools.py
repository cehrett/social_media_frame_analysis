# Helper functions

from sklearn.metrics import adjusted_rand_score
import hdbscan.prediction as hdb_predict
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# This function returns the current time as a string
def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_theories_and_embeddings_using_dict(df, data_dict, theory_col='frames', id_col='id'):
    """
    Retrieve theories and their corresponding embeddings from a dataframe and a dictionary.
    
    Parameters:
    - df (pandas.DataFrame): The dataframe containing a 'theory' column with lists of theories.
    - data_dict (dict): A dictionary mapping theories to their corresponding embeddings.
    
    Returns:
    - dict: A dictionary containing three lists: 'theories' with the theories, 'embeddings' with the embeddings, and 'id' with the IDs.
    """
    
    # Lists to store the theories, their corresponding embeddings, and IDs
    theories_with_duplicates = []
    embeddings_with_duplicates = []
    ids_with_duplicates = []

    # Iterate over each list in the 'theory' column and the 'id_col' column of the dataframe
    for sublist, post_id in zip(df[theory_col], df[id_col]):
        # Iterate over each theory in the list
        for theory in sublist:
            # If the theory has an embedding in the data_dict, append it to the lists
            if isinstance(theory,list):
                import pdb; pdb.set_trace()
            if theory in data_dict:
                theories_with_duplicates.append(theory)
                embeddings_with_duplicates.append(data_dict[theory])
                ids_with_duplicates.append(post_id)  # Append the corresponding ID
                
    # Return the results as a dictionary
    return({
        'frames': theories_with_duplicates,
        'embeddings': embeddings_with_duplicates,
        'id': ids_with_duplicates  # Add the IDs to the resulting dictionary
    })


def perform_clustering(embeddings, 
                       min_cluster_size=10, 
                       min_samples=10,
                       cluster_selection_epsilon=0,
                       umap_dim_size=50,
                      ):
    """
    Perform clustering on a list of embeddings using the HDBSCAN algorithm.
    
    Parameters:
    - embeddings (list): A list of embeddings to be clustered.
    
    Returns:
    - hdb: A fitted HDBSCAN clustering model.
    """
    
    from hdbscan import HDBSCAN
    import umap
    from collections import defaultdict

    # Step 1: Identify and extract unique embeddings
    unique_embeddings = defaultdict(list)
    for idx, emb in enumerate(embeddings):
        unique_embeddings[tuple(emb)].append(idx) # Use tuple because numpy arrays are not hashable

    unique_emb_list = list(unique_embeddings.keys())

    umap_model = umap.UMAP(n_components=umap_dim_size)
    
    # Step 2: Apply UMAP only to the unique embeddings
    reduced_unique_embeddings = umap_model.fit_transform(unique_emb_list)

    # Step 3: Map the non-unique embeddings to their UMAP-reduced counterparts
    reduced_embeddings = [reduced_unique_embeddings[unique_emb_list.index(tuple(emb))] for emb in embeddings]
    
    # Instantiate the HDBSCAN clustering model with a specified minimum cluster size
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size,
                        core_dist_n_jobs=-1,
                        prediction_data=True)
    
    # Fit the model to the embeddings
    hdb_model.fit(reduced_embeddings)
    
    # Return the fitted model
    return {'hdb':hdb_model, 'umap':umap_model}


def calculate_rand_index(cluster_1_dict, cluster_2_dict):
    """
    Compute the Rand Index between two HDBSCAN clusterings based on separate embeddings.
    
    Parameters:
    - cluster_1_dict, cluster_2_dict: dict
        Each dictionary should have:
            - 'embeddings': a list of embeddings used to fit the clustering
            - 'clustering': a fitted HDBSCAN clustering object.
    
    Returns:
    - float
        The Rand Index value between the two clusterings.
    """
    
    # Combine embeddings from both clusterings
    combined_embeddings = cluster_1_dict['embeddings'] + cluster_2_dict['embeddings']
    
    # Get cluster labels using the approximate_predict method from the HDBSCAN objects
    labels_1, strengths_1 = hdb_predict.approximate_predict(cluster_1_dict['clustering'], combined_embeddings)
    labels_2, strengths_2 = hdb_predict.approximate_predict(cluster_2_dict['clustering'], combined_embeddings)
    
    # Calculate and return the Rand Index
    return adjusted_rand_score(labels_1, labels_2)


def get_new_labels_given_clustering(clustering, new_embeddings):
    """
    Given an already-fitted clustering object and a new set of embeddings that were not included in the clustering, \
    get (approximate) cluster labels for the new embeddings (without refitting the clustering).
    
    Parameter:
    - clustering: a fitted HDBSCAN clustering model
    - new_embeddings (list): a list of new embeddings for which to get labels
    
    Returns:
    - new_labels (list): a list of (approximate) cluster labels for the new embeddings.
    """
    new_labels, _ = hdb_predict.approximate_predict(clustering, new_embeddings)
    
    return new_labels


def check_cluster_similarity(cluster1, cluster2, threshold):
    """
    Finds labels in cluster1 that are identical to labels in cluster2 based on a given threshold.

    Parameters:
    - cluster1 (np.array): Array of labels from the first cluster.
    - cluster2 (np.array): Array of labels from the second cluster.
    - threshold (float): Proportion threshold to consider two labels as identical.

    Returns:
    - dict: Dict of labels from cluster1 as keys, matching labels from cluster2 as values.
    """
    
    matching_labels = {}
    
    # Iterate through labels in cluster1
    unique_cluster1 = list(set(cluster1))
    for label1 in tqdm(unique_cluster1, desc="Processing labels from cluster1"):
        count_label1 = np.sum(cluster1 == label1)
        for label2 in set(cluster2):
            # Count instances where label1 and label2 are identical
            count_both = sum(1 for i, j in zip(cluster1, cluster2) if i == label1 and j == label2)
            # Calculate the proportion
            proportion = count_both / count_label1
            if proportion >= threshold:
                matching_labels.update({label1:label2})
                break
    
    return matching_labels


def top_n_clusters(dataframe, n=10, filters=None, exclude_minus_one=True, toptype='range', date_range=None):
    """
    Return the top n rows with the largest range of values, possibly after filtering by the provided dictionary.

    Parameters:
    - dataframe: DataFrame to analyze. Should have one row per cluster.
    - n: Number of top rows to return. Default is 10.
    - filters: Optional dict with column indices as keys and functions as values. Used to filter rows.
    - exclude_minus_one: Boolean indicating whether to exclude cluster label -1. Default is True.
    - toptype: string indicating which metric is used to select rows. Options are 'range' or 'relevance'.
    - date_range: Optional range of dates to be used. All dates outside the range will be excluded. Must be list of str in same format as dataframe columns (when latter are coerced to str).

    Returns:
    - A DataFrame containing the top n rows with the largest range of values.
    """
    # Validate input
    if toptype not in ['range', 'relevance']:
        raise ValueError("The 'toptype' parameter must be either 'range' or 'relevance'.")
    
    # Work on a copy to avoid warnings and unintended side-effects
    df_copy = dataframe.copy()
    
    # Narrow down to date_range, if provided
    if date_range:
        cols_to_keep = [col for col in df_copy.columns if date_range[0] <= str(col) <= date_range[1]]
        if toptype in df_copy.columns: cols_to_keep += [toptype]
        df_copy = df_copy[cols_to_keep]
    
    # If filters are provided
    # import pdb; pdb.set_trace()
    if filters:
        for col_index, func in filters.items():
            df_copy = df_copy[df_copy[col_index].apply(func)]
    
    # Exclude rows with cluster label -1 if the flag is set
    if exclude_minus_one:
        df_copy = df_copy[df_copy.index != -1]
    
    # Compute the range for each row and sort rows by range in descending order
    if toptype == 'range':
        df_copy = df_copy[[col for col in df_copy.columns if col not in ['range','relevance']]]
        df_copy[toptype] = df_copy.max(axis=1) - df_copy.min(axis=1)
    sorted_df = df_copy.sort_values(by=toptype, ascending=False)
    
    # Return the top n rows without the 'range'/'relevance' column
    return sorted_df.head(n).drop(columns=[toptype])

# Example usage:
# filtered_df = top_n_ranges(cluster_df, n=5, filters={0: lambda x: x > 0.5, 2: lambda x: x < 0.3})


def plot_theory_lines(df, 
                      theory_df, 
                      theory_col='frames', 
                      round_to_nearest='D', 
                      add_sum_line=False,
                      plot_title="Frame Clusters over Time Periods"
                     ):
    """
    Plot the data from df with interactive hover features linking back to the theory_df.
    
    Parameters:
    - df: The DataFrame (result from top_n_ranges).
    - theory_df: The original DataFrame with theories column.
    - theory_col: The (str) theory column name
    """
    
    # Prepare a dictionary for hover-text
    cluster_to_theory = dict(zip(theory_df['cluster_labels'], theory_df[theory_col]))
    
    # Optionally, add a line that sums the other lines' proportions
    if add_sum_line:
        # Calculate the sum of each column
        sum_row = df.sum(axis=0).to_frame().T

        # Determine the index of the new row
        new_row_index = df.index.max() + 1

        # Assign new index to sum_row
        sum_row.index = [new_row_index]

        # Concatenate sum_row to df without resetting indices
        df = pd.concat([df, sum_row])

        # Add the new key/value pair to the cluster_to_theory dictionary
        cluster_to_theory[new_row_index] = 'Sum of other rows'
    
    # Convert the columns to strings, so Plotly can use them
    if round_to_nearest=='D':
        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
    else:
        df.columns = df.columns.astype(str)
    
    # Sort the columns
    df = df.sort_index(axis=1)
    # import pdb; pdb.set_trace()
    
    # Create a new figure
    fig = go.Figure()
    
    # Loop through rows in the input dataframe
    for cluster_label, row in df.iterrows():
        hover_text = f"Cluster: {cluster_label}<br>Frame: {cluster_to_theory.get(cluster_label, 'Unknown Theory')}"
        fig.add_trace(
            go.Scatter(
                x=row.index,
                y=row.values,
                mode='lines+markers',
                name=f"Cluster {cluster_label}",
                hovertext=hover_text,
                hoverinfo="x+y+text"
            )
        )
    
    # Update layout with title, axis labels, and plot size
    fig.update_layout(
        title=plot_title,
        xaxis_title="Time Period",
        yaxis_title="Proportion",
        width=1000,
        height=600
    )
    
    return fig

# Example usage:
# plot_theory_lines(top_n_ranges(cluster_df, n=20, filters = {2: lambda x: x==0}), theory_df)


def get_embeddings_of_query_theories(username, 
                                     api_key_loc,
                                     query_theories: list[str], 
                                     skip_verification=False):
    from langchain_openai import OpenAIEmbeddings
    from .load_llm_model import prepare_to_load_model
    
    # Define API key env var
    prepare_to_load_model(username=username, api_key_loc=api_key_loc, service='openai')

    # Get embeddings of large_list elts
    if skip_verification:
        userinput = 'y'
    else:
        userinput = input("Are you sure you want to run this cell? It generates a (small) charge on OpenAI. [y/n]")
    if userinput == 'y':
        embeddings_model = OpenAIEmbeddings()
        embeddings = embeddings_model.embed_documents(query_theories)
    return embeddings


def compute_relevance(theory_df, query_theories, cluster_df):
    """
This function takes embeddings for a set of queries and returns the maximum (across that set) of average cosine similarities
of theories in each cluster to those queries. This is useful for identifying which clusters are talking about some topic of interest,
e.g., "Biden is too old".
    """
    cluster_df_out = cluster_df.copy()
    
    # Initialize a column to hold the max average similarity for each cluster
    cluster_df_out['relevance'] = 0

    for query_embedding in query_theories:
        # Compute cosine similarity for the current query_embedding
        similarities = cosine_similarity([query_embedding], list(theory_df['embeddings']))[0]

        # Assign the similarities to the theory_df for averaging
        theory_df['temp_sim'] = similarities

        # Group by cluster_labels and get average similarity
        avg_similarities = theory_df.groupby('cluster_labels')['temp_sim'].mean()

        # Update the 'relevance' column in cluster_df
        cluster_df_out['relevance'] = np.maximum(cluster_df_out['relevance'], avg_similarities)

    # Remove the temporary column
    del theory_df['temp_sim']

    return cluster_df_out


def plot_based_on_query(query_theories, theory_df, cluster_df, n=20, add_sum_line=True, filters=None, toptype='relevance', embeddings=None):
    """
    Given a list of queries (as strings), find the clusters of theories that are closest in relevance to (any of) those queries. Then plot the proportion of
    posts that fall in those clusters across time. Returns the embeddings of the query_theories and a copy of the cluster_df that contains a column
    describing the relevance of that row to the queries.
    
    Parameters:
    - query_theories: list of strings, the queries.
    - theory_df: pandas df, with each theory along with its cluster label
    - cluster_df: pandas df, with one row per cluster, and with one column for each time period. Values are proportion of posts in that cluster.
    - n: int, number of clusters
    - add_sum_line: bool, whether to plot a line that sums the others
    - filters: dict, constraints on which clusters to include, e.g., {0: lambda x: x==0} to include only clusters that don't appear in time period 0
    - toptype: whether to show the clusters with most range, or the clusters with most relevance to the queries. If toptype=='range', queries are ignored.
    - embeddings: list of lists, the embeddings of the query_theories (To save the small OpenAI charge of generating them again.) If none, generated anew.
    """
    if embeddings is None:
        embeddings = get_embeddings_of_query_theories(query_theories)
    cluster_df = compute_relevance(theory_df,embeddings,cluster_df)
    top_n = top_n_clusters(cluster_df, n=n, toptype=toptype, filters=filters)
    fig = plot_theory_lines(top_n, theory_df, add_sum_line=add_sum_line)
    
    return embeddings, cluster_df, fig


def calculate_proportions(theory_df, cluster_col='cluster_labels', time_col='time_bin'):
    """
    Calculate the share of the conversation each theory occupies in each time period.

    Parameters:
    - theory_df (pandas.DataFrame): The dataframe with 'cluster_labels' and 'time_period' columns.

    Returns:
    - pandas.DataFrame: A dataframe with proportions of conversation each theory occupies in each time period.
    """

    # Get unique values for cluster_labels and time_period
    unique_clusters = theory_df[cluster_col].unique()
    unique_periods = theory_df[time_col].unique()

    # Initialize an empty dataframe with NaNs
    cluster_df = pd.DataFrame(index=unique_clusters, columns=unique_periods)

    # Calculate proportions
    for period in unique_periods:
        time_period_df = theory_df[theory_df[time_col] == period]
        total_rows_for_period = len(time_period_df)
        for cluster in unique_clusters:
            matching_rows = len(time_period_df[time_period_df[cluster_col] == cluster])
            proportion = matching_rows / total_rows_for_period
            cluster_df.at[cluster, period] = proportion

    return cluster_df
