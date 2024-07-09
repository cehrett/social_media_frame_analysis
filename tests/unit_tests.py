import argparse
import os
import sys
import pandas as pd

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
            raise RuntimeError("Inaccurate inactive clusters")
        
        print("Test 2 - Passed")
    except Exception as e:
        error_present = True
        print("Test 2 - Failed")
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
    





