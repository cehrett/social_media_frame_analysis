# Imports
import os
import pandas as pd
import argparse
from openai import OpenAI
import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames from multiple posts using framestore')
    parser.add_argument('--prompt_loc', type=str, required=True, help='Location of the prompt template')
    parser.add_argument('--framestore_loc', type=str, required=True, help='Location of the framestore')
    parser.add_argument('--posts_loc', type=str, required=True, help='Location of the posts')
    parser.add_argument('--post_id_col', type=str, required=True, help='Name of the column in the posts dataframe that contains the post id')
    parser.add_argument('--post_text_col', type=str, required=True, help='Name of the column in the posts dataframe that contains the post text')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Name of the OpenAI model to use')
    parser.add_argument('--output_loc', type=str, required=True, help='Location to save the output')
    return parser.parse_args()

def main(args):
    # Load system prompt
    prompt_loc = args.prompt_loc
    with open(prompt_loc, 'r') as f:
        prompt = f.read()

    # Load framestore
    framestore_loc = args.framestore_loc
    framestore = pd.read_csv(framestore_loc)
    # Get text string representation of the framestore
    framestore_str = framestore.to_string(index=False)

    # Load posts
    posts_loc = args.posts_loc
    posts_full = pd.read_csv(posts_loc)
    # Narrow to only the columns we need
    posts = posts_full[[args.post_text_col]].copy()
    # Rename col text
    posts = posts.rename(columns={args.post_text_col: 'text'})
    # Deduplicate posts
    posts = posts.drop_duplicates(subset=['text'])
    # Add integer id column with header 'pid'
    posts['pid'] = range(1, 1+len(posts))
    # Put pid col first
    posts = posts[['pid', 'text']]
    # DEV
    posts = posts.head(10)
    # Get text string representation of the posts
    posts_str = posts.to_string(index=False)

    # Get OpenAI client
    client = OpenAI()

    # Create messages to send to OpenAI
    framestore_and_posts_prompt = f"""\
Below are the current framestore and the posts, expressed as csvs.

## FRAMESTORE
```{framestore_str}```

## POSTS
```{posts_str}```
"""
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": framestore_and_posts_prompt}]
    
    # Send messages to OpenAI
    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        response_format={"type": "json_object"}
    )

    # Make df of new frames
    ## First, convert the output to a dict
    output = completion.choices[0].message.content
    output_dict = json.loads(output)
    # Now, put the new frames into a df
    new_frames_df = pd.DataFrame([nf for inner_dict in output_dict['new_frames'] for nf in inner_dict['new_frames']])
    # Add the new frames to the framestore (the two dfs have the same columns)
    framestore = pd.concat([framestore, new_frames_df], ignore_index=True)
    
    # Save the updated framestore
    ## First, backup the old framestore
    ### Get current datetime as str
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    framestore_backup_loc = framestore_loc.replace('.csv', f'_backup_{now_str}.csv')
    framestore.to_csv(framestore_backup_loc, index=False)
    ## Now, save the updated framestore
    framestore.to_csv(framestore_loc, index=False)
    print(f"Updated framestore saved to {framestore_loc}")

    # Get a dataframe of the frames expressed by each post
    frames_expressed_df = pd.DataFrame(output_dict['frames'])
    # Merge with the posts df on the pid column
    posts_with_frames = pd.merge(posts, frames_expressed_df, on='pid')
    # Ensure posts_with_frames has unique 'text' values
    posts_with_frames = posts_with_frames.drop_duplicates(subset=['text'])
    # Now merge this with the posts_full df, matching the text col to the args.post_text_col col of posts_full
    posts_full_with_frames = pd.merge(posts_full, posts_with_frames, left_on=args.post_text_col, right_on='text', how='left')
    # Drop the 'text' column from the resulting DataFrame only if args.post_text_col is not 'text'
    if args.post_text_col != 'text':
        posts_full_with_frames = posts_full_with_frames.drop(columns=['text'])

    # Sanity checks: ensure that the resulting DataFrame has the same number of rows as the original posts_full df
    assert len(posts_full_with_frames) == len(posts_full), f"Length of posts_full_with_frames ({len(posts_full_with_frames)}) does not match length of posts_full ({len(posts_full)})"
    # Ensure that the resulting DataFrame has the same columns as the original posts_full df, plus the columns "pid" and "frames"
    assert set(posts_full_with_frames.columns) == set(list(posts_full.columns) + ['pid', 'frames']), f"Columns of posts_full_with_frames ({set(posts_full_with_frames.columns)}) do not match columns of posts_full plus 'frames' ({set(list(posts_full.columns) + ['frames'])})"

    # Save the posts_full_with_frames DataFrame
    import pdb; pdb.set_trace()
    posts_full_with_frames.to_csv(args.output_loc, index=False)
    print(f"Posts with frames saved to {args.output_loc}")


if __name__ == '__main__':
    args = parse_args()
    main(args)


