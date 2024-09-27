
# This curl caller code calls the API executed by 'dynamic_router_configuration_5.py'
# 
# Config 5: add, heapsort, factorial, sentiment analysis, GCD dynamic route
# Routes: Dynamic route only
#  Test Prompts: Exact match dynamic route prompts (one for each dynamic route) + Partial match prompts (one for each dynamic  route)+ 
#                Unrelated prompts (same number of route to test 'None' route classification)
#  Test this code by each of the embedding model
#
# Partha Pratim Ray
# 22 September, 2024
import requests
import json

# API endpoint
api_url = "http://localhost:5000/process_prompt"

# List of prompts for exact, partial, and no match
prompts = [
    # Exact match for dynamic route (math_operations - add function)
    "Add 4 with 9",
    # Exact match for dynamic route (heap_sort)
    "Sort the following numbers using heap sort: [9, 4, 1, 7]",
    # Exact match for dynamic route (factorial)
    "Compute factorial for 9",
    # Exact match for dynamic route (sentiment_analysis)
    "Analyze the sentiment of this text: I love FastAPI it is nice",
    # Exact match for dynamic route (factor of a number)
    "Find the GCD of 48 and 8",


    # Partial match for dynamic route (math_operations)
    "Add 5 to 10",
    # Partial match for dynamic route (heap_sort)
    "Sort following numbers [7,12,9,1]",
    # Partial match for dynamic route (factorial)
    "Find the factorial of 7",
    # Partial match for dynamic route (sentiment_analysis)
    "Find sentiment of this text: I am not happy with the service",
    # Partial match for dynamic route (factor of a number)
    "GCD of 9 and 6",

    # Unrelated prompts to test 'no_route' classification
    "Tell me about cooking techniques?",
    "What is cricketing approach?",
    "Explain the concept of soul.",
    "What is the weather like today?",
    "Who won the football match yesterday?"
]

# Function to send a POST request to the API
def send_prompt(prompt):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps({"prompt": prompt}))
        result = response.json()
        return (prompt, result)
    except Exception as e:
        return (prompt, f"Error: {str(e)}")

# Main function to execute the script
if __name__ == "__main__":
    # Loop through each prompt sequentially and send to the API
    for prompt in prompts:
        prompt_result, response = send_prompt(prompt)
        print(f"Prompt: {prompt_result}")
        print(f"Response: {response}\n")
