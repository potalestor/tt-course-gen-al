import json
import openai
import os
import pandas as pd

DATAFRAME_FILE_PATH = os.path.join("assets", "press_release_extraction.csv")
RESULT_FILE_PATH = os.path.join("assets", "parsed_press_releases.csv")
ENV_API_KEY_NAME = "OPENAI_API_KEY"
MODEL_NAME = "gpt-4o-mini"  # Default model name


def parse_press_release(pr: str, client: openai, model: str = MODEL_NAME) -> dict:
    """
    Parses a press release and extracts event information.
    :param pr: The press release text.
    :param client: An OpenAI API client instance.
    :param model: Model to use for generating summaries.
    :return: A dictionary with event data.
    """
    try:
        # Request to OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts structured data from text."
                },
                {
                    "role": "user",
                    "content": f"""
Extract structured data from the press release below. Provide the output in strict JSON format with the following keys:
- "name": The title or name of the event
- "date": The event date in DD.MM.YYYY format
- "n_participants": The number of attendees
- "n_speakers": The number of featured speakers
- "price": The ticket price or cost of attendance

If a specific piece of data is not mentioned in the press release, assign it a null value. Here is the press release text:
{pr}
"""
                }
            ],
            temperature=0  # Set temperature to 0 for deterministic responses
        )

        # Extract the response text from the OpenAI API
        summary = response.choices[0].message.content.strip()

        # Remove the ```json``` wrapper if present and parse the JSON
        summary = summary[7:-3].strip() if summary.startswith("```json") else summary
        return json.loads(summary)  # Convert the JSON string into a Python dictionary

    except Exception as e:
        # Handle any parsing errors and return default values
        print(f"Error during parsing: {e}")
        # Return a dictionary with null values for missing information
        return {key: None for key in ["name", "date", "n_participants", "n_speakers", "price"]}


def get_api_key(env_var_name: str) -> str:
    """
    Retrieves the OpenAI API key from the environment.
    """
    api_key = os.getenv(env_var_name)
    if not api_key:
        raise ValueError(f"API key not found. Set the {env_var_name} environment variable.")
    return api_key


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV file.
    """
    with open(file_path, "r", encoding="utf-8"):
        return pd.read_csv(file_path)


def save_results_to_file(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the dataframe with results to a CSV file.
    """
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Results saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def create_parsed_press_releases(client: openai, dataframe: pd.DataFrame) -> list:
    """
    Tests the `parse_press_release` function against a dataset.
    """
    # Initialize counters for testing
    correct_fields = 0
    total_fields = 0
    parsed_list = []  # Store parsed results if needed

    # Define expected fields and their types
    fields = {
        "name": str,
        "date": str,
        "n_speakers": int,
        "n_participants": int,
        "price": str
    }

    for index, row in dataframe.iterrows():
        try:
            # Parse the press release text
            parsed_release = parse_press_release(row["pr_text"], client)

            if not isinstance(parsed_release, dict):
                print(f"Error: Parsed release is not a dictionary for input {row['pr_text']}")
                continue

            # Append parsed results (optional, for later analysis)
            parsed_list.append(json.dumps(parsed_release, indent=4))

            # Compare parsed release with golden truth
            golden = json.loads(row["pr_parsed"])  # Convert JSON from CSV to dict
            for field, field_type in fields.items():
                # Ensure types are matched
                golden_field = golden[field]
                parsed_field = parsed_release.get(field)

                # Try to cast the parsed field to the expected type
                try:
                    parsed_field = field_type(parsed_field) if parsed_field is not None else None
                except (ValueError, TypeError):
                    print(f"Type mismatch for field '{field}' in release '{row['pr_text']}'")
                    parsed_field = None

                # Update counters
                total_fields += 1
                if golden_field == parsed_field:
                    correct_fields += 1
                else:
                    print(
                        f"Field mismatch for '{field}' in '{row['pr_text']}': Expected {golden_field}, Got {parsed_field}")

        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Calculate accuracy
    accuracy = (correct_fields / total_fields) * 100 if total_fields > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_fields}/{total_fields} fields correct)")

    return parsed_list


def main():
    # Load the dataset
    df = load_dataframe(DATAFRAME_FILE_PATH)

    openai.api_key = get_api_key(ENV_API_KEY_NAME)

    # Test the parser
    pp = create_parsed_press_releases(openai, df)

    # Save the results to a file
    save_results_to_file(df, RESULT_FILE_PATH)


if __name__ == "__main__":
    main()
# Output:
# Accuracy: 65.71% (23/35 fields correct)
