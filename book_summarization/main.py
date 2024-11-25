import os
import json
import nltk
import openai
import tiktoken
from typing import List

# Constants for configuration
MAX_TOKENS = 1250  # Maximum tokens for each chunk
MODEL_NAME = "gpt-4o-mini"  # Default model name
SYSTEM_MESSAGE = "You are a helpful assistant specialized in summarizing texts."
SUMMARY_PROMPT_TEMPLATE = "Please summarize the following text:\n\n{}"
DATASET_FILE_PATH = os.path.join("assets", "book_sum_dataset.txt")
ENV_API_KEY_NAME = "OPENAI_API_KEY"
SUMMARY_LENGTH_LIMIT = 953  # Name of the environment variable for API key


def split_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS, encoding_name: str = MODEL_NAME) -> List[str]:
    """
    Splits the input text into chunks based on token limits.

    :param text: Input text to split.
    :param max_tokens: Maximum number of tokens per chunk.
    :param encoding_name: Model encoding to use for token calculation.
    :return: A list of text chunks.
    """
    # Create a tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(encoding_name)

    # Break the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        # Calculate the number of tokens in the sentence
        sentence_tokens = len(encoding.encode(sentence))

        # Check if the sentence fits within the current chunk
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            # Save the current chunk and start a new one
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_tokens = sentence_tokens

    # Add any remaining text to the list of chunks
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def summarize_text(client: openai, text: str, model: str = MODEL_NAME) -> str:
    """
    Summarizes the input text using the OpenAI API.

    :param client: An OpenAI API client instance.
    :param text: The text to summarize.
    :param model: Model to use for generating summaries.
    :return: The summarized text.
    """
    try:
        # Send a request to the OpenAI chat completions endpoint
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": SUMMARY_PROMPT_TEMPLATE.format(text)},
            ]
        )
        # Extract the summarized text from the API response
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        # Handle and log any errors during the API call
        print(f"An error occurred: {e}")
        return ""


def load_dataset(file_path: str) -> dict:
    """
    Loads the dataset from the specified JSON file.

    :param file_path: Path to the JSON dataset file.
    :return: Parsed dataset as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_api_key(env_var_name: str) -> str:
    """
    Retrieves the OpenAI API key from the environment.

    :param env_var_name: Name of the environment variable holding the API key.
    :return: The API key as a string.
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"API key not found. Set the {env_var_name} environment variable.")
    return api_key


def main():
    # Load the dataset
    dataset = load_dataset(DATASET_FILE_PATH)

    # Get the OpenAI API key and initialize the client
    api_key = get_api_key(ENV_API_KEY_NAME)
    openai.api_key = api_key

    # Extract the target text from the dataset
    text = dataset["rows"][5]["row"]["chapter"]

    # Summarize the text in chunks until the entire text is summarized
    while len(text) > SUMMARY_LENGTH_LIMIT:
        chunks = split_text_by_tokens(text)
        summarized_chunks = []

        for chunk in chunks:
            summarized_text = summarize_text(openai, chunk)

            print(f"Original Chunk Length: {len(chunk)} | Summarized Chunk Length: {len(summarized_text)}")
            print("Summarized Text:", summarized_text)

            summarized_chunks.append(summarized_text)

        # Combine summarized chunks into a single text
        text = " ".join(summarized_chunks)

    # Print the final summarized text
    print("Final Summarized Text:", text)
    print("Final Summarized Text Length:", len(text))


if __name__ == "__main__":
    main()

# Output:
# Final Summarized Text:

# In this excerpt, Hawkeye, Cora, Duncan, and the Mohicans face increasing tension from mysterious noises in the forest,
# with Cora questioning whether they are supernatural or enemy-related. A familiar horse's screech offers temporary
# reassurance, but threats from wolves soon arise, causing Uncas to drive them off. As night falls, panic emerges when
# the sisters awaken in fear, leading to the accidental shooting of David, the singing-master.
# The Mohicans protect the group while Duncan searches for their canoe, and Hawkeye kills an attacker.
# They seek refuge in a chasm, where Cora worries about Duncan's safety. Hawkeye emphasizes the need to conserve
# ammunition, illustrating themes of loyalty and survival amid dwindling supplies.
# Cora insists on sending a message to Colonel Munro, choosing Uncas for the task, which highlights themes of duty and
# sacrifice as she assists Alice in navigating their dangerous situation.

# Final Summarized Text Length: 924
