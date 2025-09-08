from mongo_fetcher import fetch_filtered_messages_by_id
from testcase_extractor import extract_testcases_from_responses
from api_sender import send_context_and_save_response
from query_sequence_generator import run_api_sequence, extract_query_texts

def orchestrate_pipeline(target_id):
    # Step 1: Fetch content from MongoDB
    content = fetch_filtered_messages_by_id(target_id=target_id)

    result = send_context_and_save_response(
        content,
    )

    # Step 2: Extract test cases from saved content
    testcases = extract_testcases_from_responses(result)

    # Step 3: Send test cases to API and save response

    queries=extract_query_texts("testcases.json")
    for query in queries:
        run_api_sequence(query, "669ef7383ca0e006716c9c3802020600", num_iterations=1, csv_filename="query-sequence.csv")



if __name__ == "__main__":
    target_id=input("Enter the target ID: ")
    final_result = orchestrate_pipeline(
        target_id,
    )

