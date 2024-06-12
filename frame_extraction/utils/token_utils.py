import tiktoken

# Open source tiktoken token counter function
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

# OpenAI API only allows token counting return after prompt has been issued. To avoid unnecessary API calls, use tiktoken
def num_tokens_from_messages(messages, model, single_string=False):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        #print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613", single_string=single_string)
    elif "gpt-4" in model:
        #print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613", single_string=single_string)
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )

    if not single_string:
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    return len(encoding.encode(messages))




# Partition original messages into smaller messages and return
def partition_prompt(messages, model):
    max_tokens = 63000

    # Extract the table header from messages, assuming it is a markdown table
    table_indexes = []
    index = 0
    while index < len(messages[1]["content"]) and messages[1]["content"].find("## Table", index) != -1:
        index = messages[1]["content"].find("## Table", index)
        if (index != -1):
            table_indexes.append(index)
            index += len("## Table")

    newline_count = 0
    newline_indexes = []
    for idx in table_indexes:
        for i, char in enumerate(messages[1]["content"][idx:]):
            if char == '\n':
                newline_count += 1
                if newline_count == 3:
                    newline_index = i + idx
                    newline_indexes.append(newline_index)
    
    # Headers are between the table index and third newline index
    headers = [messages[1]["content"][table_indexes[i]:newline_indexes[i]] for i in range(len(table_indexes))]

    # If there are two markdown tables present in the code, the first one in its entirety will be used
    if len(headers) > 1:
        predefined_messages = messages

        # Split rows from second table header down
        rows = messages[1]["content"][table_indexes[1] + len(table_indexes[1]):].split('\n')

        # Predefined message is the entirety of markdown table 1 and the header table 2
        predefined_messages[1]["content"] = messages[1]["content"][table_indexes[0]:(table_indexes[1] + len(table_indexes[1]))]

        predefined_tokens = num_tokens_from_messages(predefined_messages, model)

    if len(headers) == 1:
        header = headers[0]

        # Each row is joined by '\n', so split remaining string by newline
        rows = messages[1]["content"][table_indexes[0] + len(header):].split('\n')
        # Add newline character back for accurate token count
        for i in range(len(rows)):
            rows[i] = '\n' + rows[i]

        # system prompt + markdown table header will be in every message, assuming it is a markdown table with header = header
        predefined_messages = messages
        predefined_messages[1]["content"] = header
        
        # First get system prompt and token length
        predefined_tokens = num_tokens_from_messages(predefined_messages, model)

    # If predefined tokens exceed 75% of the max_tokens throw a runtime error
    if (predefined_tokens >= max_tokens * 0.75):
        raise RuntimeError("Size of prompt and markdown table exceed token limitation")

    partitioned_messages = []

    # Now loop through rows until max_tokens is reached, then start a new prompt message
    usable_tokens = max_tokens - predefined_tokens
    current_message = ''

    for row in rows:
        
        # Get number of tokens from single row
        row_tokens = num_tokens_from_messages(row, model, True)

        # If token limit has not been reached, append row, and subtract from token limit
        if (usable_tokens - row_tokens >= 0):
            current_message += row
            usable_tokens -= row_tokens
        
        # If token limit was reached, append current message, and use current row as starting point for new message
        else:
            partitioned_messages.append(current_message)
            current_message = row
            usable_tokens = max_tokens - predefined_tokens - row_tokens
    
    # If remaining current_message is not empty, append to partitioned_messages
    if len(current_message) != 0:
        partitioned_messages.append(current_message)

    # Return a list of rows appended to header
    partitioned_user_prompt = []

    if len(headers) == 1:
        for message in partitioned_messages:
            partitioned_user_prompt.append(header + '\n' + message)

    # Return table 1 + header of table 2 + partitioned messages
    elif len(headers) > 1:
        for message in partitioned_messages:
            partitioned_user_prompt.append(messages[1]["content"][table_indexes[0]:table_indexes[1]] + '\n' + headers[1] + '\n' + message)

    return partitioned_user_prompt