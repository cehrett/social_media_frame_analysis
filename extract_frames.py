import os
import csv
import argparse
from utils.extract_theories_from_tweets import process_tweets

def main():
    parser = argparse.ArgumentParser(description="Process and save tweets.")
    parser.add_argument("input_path", help="Path to the input CSV file.")
    parser.add_argument("--results_dir", default='./', help="Directory where results (and intermediate temp file) will be written.")
    parser.add_argument("--labeled_data_path", default='./data/labeled_data.csv', help="Path for labeled data file.")
    parser.add_argument("--tweet_col", default='text', help="Name of the tweet column.")
    parser.add_argument("--api_key_loc", default='./openai_api_key.txt', help="Location of text file containing OpenAI API key.")
    parser.add_argument("--raw_csv_or_intermediate", default='c', help="Whether to use the input_path data file (c) or an intermediate file (i). Default (c). Only use (i) if previous frame extraction was interrupted before completion.")
    
    args = parser.parse_args()

    # Construct paths based on base_path and default filenames if not provided
    intermediate_path = os.path.join(args.results_dir, 'TEMP_frame_extraction.csv')
    output_path = os.path.join(args.results_dir, 'frame_extraction_results.csv')
    labeled_data_path = os.path.join('.','data', 'labeled_data.csv')
    
    try:
        df_with_theories = process_tweets(input_path=args.input_path, 
                                          tweet_col=args.tweet_col, 
                                          intermediate_path=intermediate_path, 
                                          output_path=output_path, 
                                          labeled_data_path=labeled_data_path, 
                                          chunk_size=10,
                                          raw_csv_or_intermediate=args.raw_csv_or_intermediate,
                                          api_key_loc=args.api_key_loc
                                         )
        
        df_with_theories.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Frame extraction completed successfully. Results stored in {output_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"If frames were extracted prior to error, intermediate results are stored in {intermediate_path}.")

if __name__ == "__main__":
    main()
