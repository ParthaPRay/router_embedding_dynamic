# This curl caller code calls the API executed by 'dynamic_router_configuration_1.py'
# 
# Config 1: add dynamic route
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
    # Exact match for dynamic route (math_operations)
    "Add 4 with 9",

    # Partial match for dynamic route (math_operations)
    "Add 5 to 10",

    # Completely different, no matching prompt
    "Tell me about cooking techniques?"
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
