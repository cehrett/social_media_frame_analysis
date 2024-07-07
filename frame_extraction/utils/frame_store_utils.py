import os
import pandas as pd
import random
from ast import literal_eval
import datetime

def create_frame_store(store_dir, frame_cluster_path, date):

    # First check that store_dir and cluster_path are valid
    if not os.path.exists(store_dir):
        raise FileNotFoundError(f"Store directory: {store_dir} could not be located.")
    
    if not os.path.exists(frame_cluster_path):
        raise FileNotFoundError(f"{frame_cluster_path} could not be located.")
    
    # Load frame_cluster_path as dataframe
    cluster_df = pd.read_csv(frame_cluster_path)

    # Grab unique clusters
    unique_clusters = cluster_df['cluster_labels'].unique()
    cluster_to_description = {}
    cluster_counts = {}

    # For each unique cluster, grab sample frame and description from cluster_df
    for cluster in unique_clusters:
        cluster_to_description[cluster] = cluster_df[cluster_df['cluster_labels'] == cluster]['description'].iloc[0]

        # For each cluster, get the number of times the cluster appears
        cluster_counts[cluster] = len(cluster_df[cluster_df['cluster_labels'] == cluster])
    

    # Join tables together to create store_df
    descriptions_df = pd.DataFrame(list(cluster_to_description.items()), columns=["cluster_labels", "description"])
    frequency_df = pd.DataFrame([(t, {date : v}) for t, v in cluster_counts.items()], columns=["cluster_labels", "counts"])

    store_df = pd.merge(descriptions_df, frequency_df, on="cluster_labels")

    # Finally save the dataframe as a csv in the store directory
    store_df.to_csv(os.path.join(store_dir, 'frame_store.csv'), index=False)



# Function takes in dataframe and searches all files in topic directory to find unique cluster frames
# Return modified dataframe
def populate_store_examples(store_df, root_dir, topic , n_samples):
    results_dir = os.path.join(root_dir, topic)

    # Get list of date dir from topic directory
    date_dirs = [os.path.join(results_dir, date_dir) for date_dir in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, date_dir))]

    # For each directory, access the frame_cluster_results.csv file
    for dir in date_dirs:
        if not os.path.exists(os.path.join(dir, 'frame_cluster_results.csv')):
            raise FileNotFoundError(f"frame_cluster_results.csv file not found in {dir}.")
        
        temp_cluster_df = pd.read_csv(os.path.join(dir, 'frame_cluster_results.csv'))

        # Gather only the frame column and cluster_label column
        temp_cluster_df = temp_cluster_df[['cluster_labels','frames']]

        # Group by 'cluster_labels'
        temp_cluster_df = temp_cluster_df.groupby('cluster_labels')['frames'].apply(list).reset_index()

        # If 'frames' column not present in store_df
        if 'frames' not in list(store_df.columns):
            store_df = store_df.merge(temp_cluster_df, on='cluster_labels')

        else:
            # Convert temp_cluster_df to a dictionary for faster access
            temp_cluster_dict = temp_cluster_df.set_index('cluster_labels')['frames'].to_dict()

            # If a cluster is empty for a particular date, blank string which will be removed
            for key, value in temp_cluster_dict.items():
                if value == []:
                    temp_cluster_dict[key] = ['']

            # Apply the lambda function to update the 'frames' column
            store_df['frames'] = store_df.apply(lambda x: x['frames'] + temp_cluster_dict.get(x['cluster_labels'], ['']), axis=1)


    # Remove blank string added as placeholders
    store_df['frames'] = store_df['frames'].apply(lambda x: [frame for frame in x if frame != ''])

    # Change list to set for uniqueness
    store_df['frames'] = store_df['frames'].apply(lambda x: set(x))

    # Randomly grab n_samples (if n_samples don't exist, repeat samples until n_samples reached)
    store_df['frames'] = store_df['frames'].apply(lambda x: [random.choice(list(x)) for _ in range(n_samples)])

    # Return modified dataframe
    return store_df



# Function appends the count of clusters of a particular date to the frame store
# We are assuming that the frame store has already been updated for this function...
def update_cluster_counts(store_dir, results_dir, date):
    if not os.path.exists(os.path.join(store_dir, 'frame_store.csv')):
        raise FileNotFoundError(f"Store directory: {os.path.join(store_dir, 'frame_store.csv')} could not be located.")
    
    store_df = pd.read_csv(os.path.join(store_dir, 'frame_store.csv'))

    cluster_counts = {}

    # Open frame_cluster_results.csv file
    if not os.path.exists(os.path.join(results_dir, 'frame_cluster_results.csv')):
        raise RuntimeWarning(f"frame_cluster_results.csv file not found in {results_dir}.")
    
    cluster_df = pd.read_csv(os.path.join(results_dir, 'frame_cluster_results.csv'))

    # Grab unique clusters
    unique_clusters = cluster_df['cluster_labels'].unique()

    # For each unique cluster, grab sample frame and description from cluster_df
    for cluster in unique_clusters:
        # For each cluster, get the number of times the cluster appears
        cluster_counts[cluster] = len(cluster_df[cluster_df['cluster_labels'] == cluster])
        print(cluster_counts[cluster])

    # Each row should have a dictionary, append key value pair to it
    store_df['counts'] = store_df.apply(lambda x: literal_eval(x.get('counts', '{}')).update({date : cluster_counts[x.get('cluster_labels')]}), axis=1)
    import pdb; pdb.set_trace()

    # Update and store
    store_df.to_csv(os.path.join(store_dir, 'frame_store.csv'), index=False)

# Returns a list of the inactive clusters found in the frame store
# reference_date serves as marker of the most recent date, where inactivity_period_length counts the days retroactively
def get_inactive_clusters(root_dir, topic, reference_date, inactivity_period_length, min_activity=0):
    # First load framestore and directory for clustered posts
    topic_dir = os.path.join(root_dir, topic)
    store_df = pd.read_csv(os.path.join(topic_dir, "frame_store.csv"))

    # Grab unique cluster labels from store_df
    unique_clusters = store_df['cluster_labels'].unique()

    # Create dictionary with date and cluster label counts (for each date subdirectory)
    # Get list of date dir from topic directory
    date_dirs = [os.path.join(topic_dir, date_dir) for date_dir in os.listdir(topic_dir) if os.path.isdir(os.path.join(topic_dir, date_dir))]

    # Create dataframe
    count_df = pd.DataFrame(columns=unique_clusters)
    count_df.index.name = "date"

    # For each directory, access the frame_cluster_results.csv file
    for dir in date_dirs:
        if not os.path.exists(os.path.join(dir, 'frame_cluster_results.csv')):
            raise FileNotFoundError(f"frame_cluster_results.csv file not found in {dir}.")
        
        date_df = pd.read_csv(os.path.join(dir, 'frame_cluster_results.csv'))

        # Count the number of posts from the current clustered date and add to dictionary
        count_dict = {}
        for label in unique_clusters:
            count_dict[label] = len(date_df[date_df["cluster_labels"] == label])
        
        # Add this row to the count dataframe
        count_df.loc[str(dir)[len(str(topic_dir))+1:]] = count_dict


    count_df.index = pd.to_datetime(count_df.index)

    cluster_dict = dict.fromkeys(unique_clusters, 0)
    total_files = 0

    # Calculate resulting inactivity cutoff date
    reference_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d")
    inactivity_cutoff = reference_date - datetime.timedelta(inactivity_period_length)

    inactive_clusters = []

    # For each date index, if it is between the two dates, check for the cluster activity
    for index, row in count_df.iterrows():
        if index >= inactivity_cutoff and index <= reference_date:
            # If value for a cluster is less than inactivity_value, +1 to associated value
            for label in cluster_dict.keys():
                if int(row[label]) <= min_activity:
                    cluster_dict[label] += 1

            total_files += 1

    # If cluster is marked 'inactive' for all files satisfying the range, it is inactive
    for cluster in cluster_dict.keys():
        if cluster_dict[cluster] == total_files:
            inactive_clusters.append(cluster)

    # Return the list of inactive_clusters
    return inactive_clusters