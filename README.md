# Social media frame extraction and analysis

## Overview

Social media posts may often be described as expressing some *frame*, where a frame is defined as a declarative claim about the world.
For example, a tweet such as "I see people saying Biden is too old to run -- would you rather have a complete political noob running the most complex government and economy in history?" could be seen as expressing the frames "Joe Biden is not too old to be President" and "Experience is an important qualification for political office".

Examining the frames expressed by social media posts provides a valuable lens for analysis. A particular frame may, for example, be trending even when no particular phrasing or expression of that frame is. Similarly, analyzing posts at the frame-level can help expose groups that coordinate to promote or demote a particular frame in public attention.

Manually labeling posts for their frame content is laborious. This repository includes tools for using large language models (LLMs) to extract frames from social media posts. Broadly, the code tools here automate embedding the social media posts into a text prompt which is input into an LLM, along with instructions for the LLM to identify any frames present in the posts. The LLM then produces output in the form of a list of frames found in each post. This list may be empty; e.g., the post "Whoa! Anyone see that???" does not appear to express any frame.

Tools in this repository include the following:

    * Frame extraction script: Extracts frames from each social media post in a data set.
    * Frame embedding script: Converts the frames found in the frame extraction step into numerical embeddings.
    * Frame clustering script: Clusters frames together, so that close paraphrases are gathered into a single cluster. E.g., the LLM might label one post as expressing "Joe Biden is too old to be president" and another as "Biden is too old to run for president"; the clustering step should identify that these two frames are part of a single frame-cluster.
    * Analysis dashboard notebook: A Jupyter notebook that facilitates the analysis of the frame-clusters with respect to other variables of interest in the data.
    * Bayesian account clustering script: Clusters accounts together, using two sources of information: flags (weak markers of unusual account status, e.g. indications of potential account inauthenticity) and frames (the frame-clusters expressed by each account's posts, as found by the frame-extraction, frame-embedding, and frame-clustering scripts above).
    
The rest of this README file will describe each of the above three sets of tools, along with documentation about how to run them.

## Frame extraction script

This script takes as input a data set of social media posts and returns as output a list, for each post, of frames found (by the LLM) in the post. The input is expected to be a .csv file containing (at least) a column labeled 'text', which is the column which will be used by the script for frame extraction, and a column 'id', which should be a unique identifier for each post. The output of the script is a file 'frames.csv' containing two columns: 'id', and 'frames'.

The script uses the OpenAI API for frame extraction. For this, you must have an OpenAI account. Using this script will incur charges on your OpenAI account. Your OpenAI API key should be placed in the file `./openai_api_key.txt`, or specified via the optional input `--api_key_loc`.

If your data set is saved in `/loc/data.csv`, then the script may be invoked as `python extract_frames.py /loc/data.csv`. The script's output will be stored in `./frame_extraction_results.csv`.

Note that by default the script uses GPT3.5. Note also that the script de-deduplicates the tweets, so that OpenAI is only queried to produce an embedding for each unique frame. The final output reduplicates the data to your original data structure; the deduplication is only to avoid (costly) extraneous calls to the OpenAI LLM.

The script uses a prompt template found in `/utils/prompt_template.txt`. If you wish to customize your prompt template, you may edit that file.

If you have hand-labeled examples (anywhere from 5 to hundreds), you can use these to improve the model's prompt through few-shot learning. Provide your examples as a csv via the optional input argument `--labeled_examples`. Your labeled example csv should have a column `text` and a column `frames`.

## Frame embedding

This script converts the frames found in the frame extraction step into numerical embeddings. OpenAI services are used for this; hence, your OpenAI API key must be made available at `./openai_api_key.txt`, or specified via the optional input `--api_key_loc`. Using this script will incur a charge on your OpenAI account.

In order to run this, you must first have a csv file of data (as output by the frame extraction script) that has at least a column `frames`, containing (string representations of) lists of strings.

Note that the script de-deduplicates the frames, so that OpenAI is only queried to produce an embedding for each unique frame.

The script will store the embeddings at `data/frame_embeddings.json`.

## Frame clustering

This script clusters the frames found in the frame extraction step, with the intention of providing cluster labels such that each cluster contains frames which are rough paraphrases of each other. Note that the clustering tends not to be sensitive to negation, and thus "Joe Biden is too old to be president" and "Joe Biden is not too old to be president" may be clustered together. Thus the frame-clusters may be said to represent *topics* rather than frames.

The script uses a two-step clustering process. First, these embeddings are dimension-reduced (using UMAP) to 50 dimensions. Second, the dimension-reduced embeddings are clustered using HDBScan. Unlike the frame extraction step (which queries OpenAI to run the LLM), all of this computation is performed locally.

This script requires as input the file `frame_extraction_results.csv` produced by the frame extraction script. It produces as output a file `frame_cluster_results.csv` with two columns: 'id' and 'cluster_label'.

## Analysis dashboard notebook

This Jupyter notebook facilitates analysis of the frame-clusters, particularly in relation to sentiment and to other binary variables of interest present in your data. E.g., if some accounts are known to come from a coordinated disinformation campaign, this notebook would facilitate the comparison of those accounts' posts with the other posts in the data.

The notebook includes instructions guiding its use.

## Baysian account clustering

This script uses any available (potentially weak) markers of unusual account status, along with frame-clusters expressed in the posts, to cluster the accounts.

To run this script, you will need:

1. A csv containing the top unusual/suspicious frame-clusters, as output by the script `find_suspicious_frame_clusters.py`.
2. A csv containing the author ids, flags, and number of posts, as output by the script `find_suspicious_frame_clusters.py`.
3. A csv containing the frame-cluster usage information per each account, as output by the script `find_suspicious_frame_clusters.py`.