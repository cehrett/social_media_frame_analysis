import argparse
import os
import sys
import pandas as pd
import shutil
import ast

sys.path.append(r'C:\Users\coope\InternshipCode\social_media_frame_analysis')
# Script to test functionalities with sample data
parent_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(parent_dir, 'original_data')
results_dir = os.path.join(parent_dir, 'modified_data')
requisite_dir = os.path.join(parent_dir, 'requisite_data')

# Run these tests unless specified otherwise (in command line arguments)
parser = argparse.ArgumentParser(description="Run unit tests on sample data")
parser.add_argument("--test_list", default="all", help="Input unit tests to run separated by ','")
parser.add_argument("--api_key_loc", default='./openai_api_key.txt', help="Location of text file containing OpenAI API key.")
parser.add_argument("--clear_output_data", default=True, action='store_false', help="Delete all data produced by test script once complete.")

args = parser.parse_args()
if args.test_list != "all":
    test_list = args.test_list.split(',')

    for i in range(len(test_list)):
        test_list[i] = int(test_list[i])

# If error present in the test cases, do not remove generated data
error_present = False

# Context manager to suppress print statements from pre-defined functions
class SuppressStdout:
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout
        self.devnull.close()
        if exc_type is not None:
            print(f"An exception occurred: {exc_value}")
        return False  # Propagate the exception if any
    

'''--- UNIT TESTS BEGIN ---'''

# Unit test 1, frame extraction test
from frame_extraction.extract_frames import process_and_save_posts
if args.test_list == "all" or 1 in test_list:
    try:
        with SuppressStdout():
            process_and_save_posts(input_path=os.path.join(data_dir, "example_frames.csv"),
                                    results_dir=results_dir,
                                    labeled_data_path=os.path.join(requisite_dir,"labeled_data.csv"),
                                    text_col="text",
                                    api_key_loc=args.api_key_loc,
                                    raw_csv_or_intermediate='c',
                                    system_prompt_loc=os.path.join(requisite_dir, "oai_system_message_template.txt")
                                )
        
        # Check for presence of "frame_extraction_results.csv", and ensure data entries are "viable"
        frame_df = pd.read_csv(os.path.join(results_dir, "frame_extraction_results.csv"))
        
        if frame_df['frames'].isnull().any():
            raise ValueError("Returned null value for frame extraction.")

        print("Test 1 - Passed")
    except Exception as e:
        error_present = True
        print("Test 1 - Failed")
        print(e)


# Unit test 2, inactivity testing
from frame_extraction.utils.frame_store_utils import get_inactive_clusters
if args.test_list == "all" or 2 in test_list:
    try:
        # Test no activity
        inactive_clusters_sub_1 = get_inactive_clusters(root_dir=data_dir,
                                                  topic='inactiveClusterData', 
                                                  reference_date='2024-08-04', 
                                                  inactivity_period_length=3, 
                                                  min_activity=0
        )

        # Test at most '5' activity
        inactive_clusters_sub_2 = get_inactive_clusters(root_dir=data_dir,
                                                  topic='inactiveClusterData', 
                                                  reference_date='2024-08-04', 
                                                  inactivity_period_length=3, 
                                                  min_activity=5
        )

        # sub test 1 should only contain the cluster 6 and 2 should contain 4,5,6
        if inactive_clusters_sub_1 != [6] or inactive_clusters_sub_2 != [4,5,6]:
            raise RuntimeError("Inaccurate inactive clusters.")
        
        print("Test 2 - Passed")
    except Exception as e:
        error_present = True
        print("Test 2 - Failed")
        print(e)

from frame_extraction.utils.frame_store_utils import create_frame_store
if args.test_list == "all" or 3 in test_list:
    try:
        # Test frame store creation
        create_frame_store(store_dir=results_dir, frame_cluster_path=os.path.join(data_dir, "FrameStoreData","2024-08-04","frame_cluster_results.csv"), date="2024-08-04")

        frame_store_created = pd.read_csv(os.path.join(results_dir, "frame_store.csv"))
        frame_store_original = pd.read_csv(os.path.join(data_dir,"FrameStoreData", "frame_store.csv"))

        frame_store_created.to_csv(os.path.join(results_dir, "frame_store_created.csv"), index=False)

        # Verify output
        if not frame_store_created.equals(frame_store_original):
            raise RuntimeError("Frame store output does not match intended output.")

        print("Test 3 - Passed")
    except Exception as e:
        error_present = True
        print("Test 3 - Failed")
        print(e)

from frame_extraction.utils.frame_store_utils import populate_store_examples
if args.test_list == "all" or 4 in test_list:
    try:
        # Test population of frame store with examples
        unpopulated_frame_store = pd.read_csv(os.path.join(data_dir, "FrameStoreData", "frame_store.csv"))
        populated_frame_store = populate_store_examples(unpopulated_frame_store, root_dir=data_dir, topic="FrameStoreData", n_samples=3)

        # Test populated frame store against correct frame store
        populated_frame_store_original = pd.read_csv(os.path.join(data_dir, "FrameStoreData", "populated_frame_store.csv"))

        populated_frame_store.to_csv(os.path.join(results_dir, "populated_frame_store.csv"), index=False)

        # Analyze row equivalancy in produced frames
        for index, row in populated_frame_store.iterrows():
            if ast.literal_eval(populated_frame_store_original.iloc[index]['frames']) != row['frames']:
                RuntimeError("Populated frame store output does not match intended output.")
        
        print("Test 4 - Passed")
    except Exception as e:
        error_present = True
        print("Test 4 - Failed")
        print(e)

from frame_extraction.utils.frame_store_utils import update_cluster_counts
if args.test_list == "all" or 5 in test_list:
    try:
        # Copy frame store to results dir, update and check
        if not os.path.exists(os.path.join(results_dir, "frame_store.csv")):
            shutil.copy(os.path.join(data_dir, "FrameStoreData", "frame_store.csv"), os.path.join(results_dir, "frame_store.csv"))

        # Test the updating of cluster counts
        update_cluster_counts(store_dir=results_dir,
                              results_dir=os.path.join(data_dir, "FrameStoreData", '2024-08-05'),
                              date='2024-08-05')
        
        # Compare to correct updated frame store
        created_updated_frame_store = pd.read_csv(os.path.join(results_dir, "frame_store.csv"))
        original_updated_frame_store = pd.read_csv(os.path.join(data_dir, "FrameStoreData", "multiday_frame_store.csv"))

        # Rename copied file for identification purposes
        if os.path.exists(os.path.join(results_dir, "multiday_frame_store.csv")):
            os.unlink(os.path.join(results_dir, "multiday_frame_store.csv"))

        os.rename(os.path.join(results_dir, "frame_store.csv"), os.path.join(results_dir, "multiday_frame_store.csv"))

         # Analyze row equivalancy in produced frames
        for index, row in original_updated_frame_store.iterrows():
            if ast.literal_eval(original_updated_frame_store.iloc[index]['counts']) != ast.literal_eval(row['counts']):
                raise RuntimeError("Updated frame store output does not match intended output.")

        print("Test 5 - Passed")
    except Exception as e:
        error_present = True
        print("test 5 - Failed")
        print(e)

'''--- UNIT TESTS END ---'''

# At the end of the tests, clear all of the newly created data (if all test cases pass)
if not error_present and args.clear_output_data:
    # Traverse the directory tree
    for root, dirs, files in os.walk(results_dir, topdown=False):
        # Remove each file
        for file in files:
            file_path = os.path.join(root, file)
            os.unlink(file_path)
            print(f"Removed file: {file_path}")

        # Remove each directory
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

    print(f"All contents removed from {results_dir}")
    





