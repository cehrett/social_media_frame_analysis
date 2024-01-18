import pandas as pd
import numpy as np
import os
import ast
import argparse
import string
import json
from langchain.embeddings import OpenAIEmbeddings
from utils.load_llm_model import prepare_to_load_model


def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return np.nan


def main():
    parser = argparse.ArgumentParser(description="Get numerical embeddings for frames.")
    parser.add_argument("--embeddings_path", 
                        default=os.path.join('.', 'data', 'frame_embeddings.json'), 
                        help="Location to store embeddings.")
    parser.add_argument("--frames_path", 
                        default=os.path.join('.', 'frame_extraction_results.csv'), 
                        help="Location of frames to be embedded.")
    parser.add_argument("--api_key_loc", 
                        default='./openai_api_key.txt', 
                        help="Location of text file containing OpenAI API key.")
    
    args = parser.parse_args()
    
    df_with_frames = pd.read_csv(args.frames_path)
    df_with_frames.frames = df_with_frames.frames.apply(safe_literal_eval)
    df_with_frames.frames = \
        df_with_frames.frames.apply(
            lambda x: [elt.strip(string.whitespace + string.punctuation) \
                for elt in x if elt is not None] if isinstance(x, list) else x)

    # import pdb; pdb.set_trace()
    # Get large list of all theories
    large_list = \
        [element for sublist in df_with_frames['frames'] \
            for element in sublist if element is not None]
    large_list = list(set(large_list)) # Filter to unique elts

    # Define API key env var and initialize model
    prepare_to_load_model(service='openai', api_key_loc=args.api_key_loc)
    embeddings_model = OpenAIEmbeddings()
    
    try:
        # Get embeddings of large_list elts
        embeddings = embeddings_model.embed_documents(large_list)

        # Creating a dictionary
        embeddings_dict = dict(zip(large_list, embeddings))

        # Saving the dictionary to a JSON file
        with open(args.embeddings_path, 'w') as file:
            json.dump(embeddings_dict, file)
        print(f"Frame embedding completed successfully. Results stored in {args.embeddings_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

if __name__ == "__main__":
    main()
