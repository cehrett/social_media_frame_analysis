# Imports
from langchain import PromptTemplate
import os

# Import template text
# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Build the path to the file
oai_system_message_file_path = os.path.join(dir_path, 'oai_system_message_template.txt')
# Now use this path to open the file
with open(oai_system_message_file_path, 'r') as file:
    oai_system_message_template = file.read()
    

def make_prompt_for_oai_narr_extr(post, 
                                  examples, 
                                  system_template=oai_system_message_template,
                                 ):
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    """
    Examples should be given as a list of dicts each with two k/v pairs: post and label.
    """
    
    # Initialize messages list
    messages = []
    
    # Add system prompt
    messages.append(SystemMessage(content=system_template))
    
    # Add each example
    for example in examples:
        messages.append(HumanMessage(content=example['post'], example=True))
        messages.append(AIMessage(content=example['label'], example=True))
    
    messages.append(HumanMessage(content=post, example=False))
    
    return messages


def load_labeled_examples(df,
                          num_examples_each_type=3,
                          nonestring='["None"]'
                         ):
    import pandas as pd
    
    # Split the DataFrame into two based on 'label' value
    df_none = df[df['label'] == nonestring]
    df_not_none = df[df['label'] != nonestring]

    # Sample 3 rows from each DataFrame
    sample_none = df_none.sample(num_examples_each_type)
    sample_not_none = df_not_none.sample(num_examples_each_type)

    # Concatenate the results
    sampled_df = pd.concat([sample_none, sample_not_none])

    # Shuffling the DataFrame
    sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)

    # Converting the DataFrame to a list of dictionaries
    list_of_dicts = sampled_df[['post','label']].to_dict('records')
    return list_of_dicts