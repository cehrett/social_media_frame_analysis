import argparse
import os
import sys
import pandas as pd

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

args = parser.parse_args()
if args.test_list != "all":
    test_list = args.test_list.split(',')


# Unit test 1, frame extraction test
from frame_extraction.extract_frames import process_and_save_posts
if args.test_list == "all" or 1 in test_list:
    try:
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
        print("Test 1 - Failed")
        print(e)




# At the end of the tests, clear all of the newly created data

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
    





