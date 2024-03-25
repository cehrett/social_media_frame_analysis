import os
import csv
import traceback
from utils.extract_theories_from_tweets import process_tweets

def process_and_save_tweets(input_path, 
                            results_dir='./', 
                            labeled_data_path='./data/labeled_data.csv', 
                            text_col='text', 
                            api_key_loc='./openai_api_key.txt', 
                            raw_csv_or_intermediate='c',
                            system_prompt_loc='./utils/oai_system_message_template.txt'
                           ):
    intermediate_path = os.path.join(results_dir, 'TEMP_frame_extraction.csv')
    output_path = os.path.join(results_dir, 'frame_extraction_results.csv')
    labeled_data_path = os.path.join('.','data', 'labeled_data.csv')
    
    try:
        df_with_frames = process_tweets(input_path=input_path,
                                        tweet_col=text_col,
                                        intermediate_path=intermediate_path, 
                                        output_path=output_path, 
                                        labeled_data_path=labeled_data_path, 
                                        chunk_size=10,
                                        raw_csv_or_intermediate=raw_csv_or_intermediate,
                                        api_key_loc=api_key_loc,
                                        system_prompt_loc=system_prompt_loc
                                       )
        
        df_with_frames.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        os.remove(intermediate_path)
        print(f"Frame extraction completed successfully. Results stored in {output_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()  # Print the full traceback
        print(f"If frames were extracted prior to error, intermediate results are stored in {intermediate_path}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process and save tweets.")
    parser.add_argument("input_path", help="Path to the input CSV file.")
    parser.add_argument("--results_dir", default=os.path.join('.', 'output'), help="Directory where results (and intermediate temp file) will be written.")
    parser.add_argument("--labeled_data_path", default=os.path.join('.', 'data', 'labeled_data.csv'), help="Path for labeled data file.")
    parser.add_argument("--text_col", default='text', help="Name of the text column.")
    parser.add_argument("--api_key_loc", default=os.path.join('.', 'openai_api_key.txt'), help="Location of text file containing OpenAI API key.")
    parser.add_argument("--raw_csv_or_intermediate", default='c', help="Whether to use the input_path data file (c) or an intermediate file (i). Default (c). Only use (i) if previous frame extraction was interrupted before completion.")
    parser.add_argument("--system_prompt_loc", default=os.path.join('.', 'utils', 'oai_system_message_template.txt'), help="System prompt to give to LLM when extracting frames.")
    args = parser.parse_args()

    process_and_save_tweets(input_path=args.input_path,
                            results_dir=args.results_dir,
                            labeled_data_path=args.labeled_data_path,
                            text_col=args.text_col,
                            api_key_loc=args.api_key_loc,
                            raw_csv_or_intermediate=args.raw_csv_or_intermediate,
                            system_prompt_loc=args.system_prompt_loc
                           )
