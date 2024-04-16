import pandas as pd
import ast

def extract_output_narrs(results, prompt):
    """
    Take in the raw output from the LLM, and return a list of string representations of lists of strings.
    """
    # Get the generated text, excluding the prompt
    narrs = [result['generated_text'][len(prompt):] for result in results]
    
    # Only keep up until the first close bracket
    narrs = [narr.split(']')[0] for narr in narrs]
    
    return narrs


def convert_to_list(s):
    """
    Take in a string representation of a list of strings, convert it to a true list of strings.
    """
    if s.strip() == 'None':
        return []
    
    # Remove leading and trailing whitespace, remove newlines
    s = s.strip().replace('\n', '')
    
    # Make sure every item in the list is wrapped in quotation marks
    elements = s.split(',')
    cleaned_elements = ['"' + element.strip(' \n\"\t') + '"' for element in elements]
    s = ','.join(cleaned_elements)
    
    # Add brackets to make it a valid list representation
    s = '[' + s + ']'
    
    try: list_rep = ast.literal_eval(s)
    except: list_rep = []
    return list_rep


def convert_to_series(results, prompt):
    """
    Take in raw output from the LLM, and return a pandas series of lists of strings.
    """
    # Extract the narratives as a list of string representations of lists of strings
    narrs = extract_output_narrs(results, prompt)
    
    # Converting each string representation to an actual list
    list_rep = [convert_to_list(narr) for narr in narrs]

    # Creating a pandas Series from the list of lists
    series_rep = pd.Series(list_rep)
    
    return series_rep


def convert_oai_output_to_series(results):
    """
    Take in raw output from the LLM, and return a pandas series of lists of strings.
    """
    # Extract the narratives as a list of string representations of lists of strings
    narrs = extract_output_narrs(results, prompt)
    
    # Converting each string representation to an actual list
    list_rep = [convert_to_list(narr) for narr in narrs]

    # Creating a pandas Series from the list of lists
    series_rep = pd.Series(list_rep)
    
    return series_rep