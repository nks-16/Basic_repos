import json
import re

def extract_testcases_from_responses(raw_response):
    formatted_output_file="testcases.json"
    # Load raw response data :
    entry = raw_response

    formatted_testcases = []

    try:
            content = entry["response"]["openAIResponse"]["choices"][0]["message"]["content"]
            # Extract JSON block using regex
            match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
            if match:
                raw_json_text = match.group(1)
                cleaned_json_text = raw_json_text.replace("\\n", "\n").replace('\\"', '"')
                testcases = json.loads(cleaned_json_text)
                formatted_testcases.extend(testcases)
            else:
                print("No JSON block found in content")
    except Exception as e:
            print(f"Error processing entry: {e}")

    # Save formatted test cases
    with open(formatted_output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_testcases, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(formatted_testcases)} test cases to {formatted_output_file}")
    return formatted_testcases
