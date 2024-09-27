# Multi-shot, dynamic routing
# THRESHOLD value based comparison is included for "no_route"
# Multiple math functions with schema
# LLM is not called for "no_route"
# Both static and dynamic routes are selected
# 'route_selection_duration' is included in csv in ns to measure time of selecting a route 
# 'llm_invoked' is included in csv to check whether llm is invoked, 1 means yes, 0 means no, useful for dyanmic routing analysis
###########
# Configuration: 1: add function of dynamic route
###########
# Ask the code by follows: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "What is 3 raised to the power of 2?"}'
# Embed model:                 |        Threshold: 
# all-minilm:33m               |          0.4950
# nomic-embed-text             |          0.63
# snowflake-arctic-embed:110m  |          0.66
# mxbai-embed-large            |          0.62
# Partha Pratim Ray
# 20 September, 2024


from fastapi import FastAPI
import numpy as np
import requests
import threading
import psutil
import time
import csv
import os
from pydantic import BaseModel
from queue import Queue
from statistics import mean
import json

app = FastAPI()

# Define the threshold for similarity score 
THRESHOLD = 0.62  # Adjust based on embedding model

# Define the embedding model and LLM
embed_model = "mxbai-embed-large"  # Embedding model
model_name = "qwen2:0.5b-instruct"  # LLM for dynamic routes
OLLAMA_API_URL = "http://localhost:11434/api/embed"
OLLAMA_LLM_URL = "http://localhost:11434/api/chat"

# CSV file setup
csv_file = 'dynamic_static_router_logs.csv'
csv_headers = [
    'timestamp', 'model_name', 'embed_model', 'prompt', 'response', 'route_type', 'route_selected',
    'semantic_similarity_score', 'similarity_metric', 'vector', 'total_duration', 
    'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 
    'eval_duration', 'tokens_per_second', 'avg_cpu_usage_during', 'memory_usage_before', 
    'memory_usage_after', 'memory_allocated_for_model', 'network_latency', 'total_response_time', 'route_selection_duration', 'llm_invoked'
]

csv_queue = Queue()
cpu_usage_queue = Queue()
memory_usage_queue = Queue()
is_monitoring = False
memory_allocated_for_model = 0

# CSV writer thread to log the data
def csv_writer():
    while True:
        log_message_csv = csv_queue.get()
        if log_message_csv is None:  # Exit signal
            break
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(csv_headers)
            writer.writerow(log_message_csv)

csv_thread = threading.Thread(target=csv_writer)
csv_thread.start()

class Prompt(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    # Measure memory usage for the model
    global memory_allocated_for_model
    memory_allocated_for_model = load_model_and_measure_memory(model_name)
    print(f"Memory Allocated for Model: {memory_allocated_for_model / (1024)} MB")

# Function to measure memory allocated after loading the model
def load_model_and_measure_memory(model_name):
    process = psutil.Process()

    # Measure memory before loading the model
    memory_before_loading = process.memory_info().rss

    # Load the model (using an empty prompt to just load the model)
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False
    }
    response = requests.post(OLLAMA_LLM_URL, json=payload)
    if response.status_code == 200:
        print(f"Model {model_name} loaded successfully.")
    else:
        print(f"Failed to load model {model_name}")

    # Measure memory after loading the model
    memory_after_loading = process.memory_info().rss
    memory_allocated = memory_after_loading - memory_before_loading

    return memory_allocated

# Modified Route class to include functions and function_schemas
class Route:
    def __init__(self, name, utterances, responses=None, dynamic=False, function_schemas=None, functions=None):
        self.name = name
        self.utterances = utterances
        self.responses = responses  # Predefined static responses
        self.dynamic = dynamic
        self.function_schemas = function_schemas or []  # For dynamic routes
        self.functions = functions or []  # For dynamic routes

# Define functions for dynamic routes
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Define function schemas
add_schema = {
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number."
            },
            "b": {
                "type": "number",
                "description": "The second number."
            }
        },
        "required": ["a", "b"]
    }
}

# Define routes with static and dynamic options
routes = [
    Route(
        name="physics",
        utterances=[
            "What is Newton's first law of motion?",
            "Can you explain the theory of relativity?",
            "What is quantum mechanics?"
        ],
        responses=[
            "Newton's first law states that an object will remain at rest or in uniform motion in a straight line unless acted upon by an external force.",
            "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
            "Quantum mechanics studies subatomic particles and their interactions."
        ]
    ),
    Route(
        name="biology",
        utterances=[
            "What is the process of photosynthesis?",
            "Can you explain the structure of DNA?",
            "What is cellular respiration?"
        ],
        responses=[
            "Photosynthesis is how green plants use sunlight to synthesize foods from carbon dioxide and water.",
            "DNA consists of two strands that form a double helix, carrying genetic information.",
            "Cellular respiration converts glucose and oxygen into energy, carbon dioxide, and water."
        ]
    ),
    Route(
        name="math_operations",
        utterances=[
            # Combined utterances for multiple functions                   
            "Add x and y",
            "What is the sum of 3 and 5?",
            "Add 4 with 9",
            "Summation of 2 with 9",
            "Add 4 and 9"
            
        ],
        dynamic=True,  # Dynamic route that triggers LLM
        functions=[add],
        function_schemas=[add_schema]
    )
]

# Function to create multi-shot prompt for the LLM
# During testing comment necessary parts for zero-shot, one-shot and few shot in same number of examples for each functions
# For zero-shot comment all examples except the last "Now,..."
# For one-shot comment all examples except the last "Now,..." and any one example for each functions
# For few-shot comment all examples except the last "Now,..." and all three examples for each functions
def create_multi_shot_prompt(prompt: str, route: Route) -> str:
    examples = """
Example 1:
User: Add 5 with 7
Assistant:
Function: add
Arguments: {"a": 5, "b": 7}
Result: 12

Example 2:
User: Sum of 9 and 8
Assistant:
Function: add
Arguments: {"a": 9, "b": 8}
Result: 17

Example 3:
User: What is 5 plus 7?
Assistant:
Function: add
Arguments: {"a": 5, "b": 7}
Result: 12

Now, respond to this query:
User: """ + prompt + """
Assistant:
"""
    return examples

# Resource monitoring thread to track CPU and memory usage
def monitor_resources():
    global is_monitoring
    process = psutil.Process()
    while is_monitoring:
        cpu_usage = psutil.cpu_percent(interval=0.01)
        memory_usage = process.memory_info().rss  # Memory in bytes
        cpu_usage_queue.put(cpu_usage)
        memory_usage_queue.put(memory_usage)
        time.sleep(0.01)  # Poll every 10ms

# Function to call LLM with multi-shot prompt and extract metrics
def call_llm_with_multi_shot(prompt, route):
    response = requests.post(
        OLLAMA_LLM_URL,
        json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "stream": False}
    )
    
    response_json = response.json()

    # Check if the expected keys are present in the response
    if 'message' in response_json and 'content' in response_json['message']:
        generated_response = response_json['message']['content']
    else:
        generated_response = "Error: Unexpected response structure from LLM"

    # Parse the LLM's response to extract function calls and arguments
    function_calls = parse_llm_response(generated_response)
    response_texts = []
    
    # Process each function call in sequence
    for function_name, arguments in function_calls:
        if function_name and arguments:
            # Find the function in the route
            function = next((f for f in route.functions if f.__name__ == function_name), None)
            if function:
                # Call the function with the arguments
                try:
                    result = function(**arguments)
                    response_texts.append(f"Function: {function_name}\nResult: {result}")
                except Exception as e:
                    response_texts.append(f"Function: {function_name}\nError executing function: {e}")
            else:
                response_texts.append(f"Function {function_name} not found.")
        else:
            response_texts.append(f"Could not parse function call from LLM response.")

    # Combine results from all tasks
    response_text = '\n'.join(response_texts)

    # Extract the relevant metrics
    total_duration = response_json.get('total_duration', 0)
    load_duration = response_json.get('load_duration', 0)
    prompt_eval_count = response_json.get('prompt_eval_count', 0)
    prompt_eval_duration = response_json.get('prompt_eval_duration', 0)
    eval_count = response_json.get('eval_count', 0)
    eval_duration = response_json.get('eval_duration', 1)  # Avoid division by zero

    # Return the response and extracted metrics
    return {
        "generated_response": response_text,
        "metrics": {
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration
        }
    }


# Function to parse LLM response
# Function to parse LLM response and map to valid functions
def parse_llm_response(response_text):
    # Extract multiple function calls from the LLM response
    lines = response_text.strip().split('\n')
    function_calls = []
    function_name = None
    arguments = None

    # Map alternative terms to the correct functions
    function_mapping = {
        "add": "add"
    }

    for line in lines:
        if line.startswith('Function:'):
            if function_name and arguments:
                function_calls.append((function_name, arguments))
            raw_function_name = line[len('Function:'):].strip()
            # Map function to the defined functions
            function_name = function_mapping.get(raw_function_name.lower(), None)
            arguments = None
        elif line.startswith('Arguments:'):
            args_text = line[len('Arguments:'):].strip()
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                arguments = None
    
    if function_name and arguments:
        function_calls.append((function_name, arguments))  # Append the last function call

    return function_calls


# Main route processing API endpoint
@app.post("/process_prompt")
async def process_prompt(request: Prompt):
    global is_monitoring
    start_time = time.time()
    data = request.dict()
    prompt = data['prompt']
    
    llm_invoked = 0  # Initialize llm_invoked to 0 to know whether llm is invoked or not, useful for dynamic routing

    try:
        # Start resource monitoring
        is_monitoring = True
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Capture the start time for route selection
        route_start_time = time.time()

        # Get embedding for the prompt
        prompt_embedding, embed_metrics = get_embedding(prompt)

        # Find the best route (Static or Dynamic) based on the prompt
        best_route, similarity = find_best_route(prompt_embedding, routes)

        # Calculate the time taken for route selection
        route_selection_duration = (time.time() - route_start_time) * 1e9  # Convert to nanoseconds

        # Check if the similarity score is below the threshold
        if similarity < THRESHOLD:
            best_route = None

        if best_route is None:
            print("No matching route found.")
            route_name = "no_route"
            response = "No route found"
            route_type = "none"
            route_selected = route_name
            # Default values for missing metrics in case no route is found
            prompt_eval_duration = 0
            eval_count = 0
            eval_duration = 0
            prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
            total_duration = embed_metrics.get('total_duration', 0)
            load_duration = embed_metrics.get('load_duration', 0)
            print(f"No Route Response: {response}")  # Debugging print statement
        else:
            print(f"Selected Route: {best_route.name} with similarity: {similarity}")
            route_name = best_route.name
            route_selected = route_name
            route_type = "dynamic" if best_route.dynamic else "static"

            if best_route.dynamic:
                llm_invoked = 1  # Set llm_invoked to 1 since LLM is called
                # For dynamic routes, trigger LLM with multi-shot prompt
                multi_shot_prompt = create_multi_shot_prompt(prompt, best_route)
                llm_response = call_llm_with_multi_shot(multi_shot_prompt, best_route)
                response = llm_response['generated_response']
                # Extract the metrics for dynamic routes
                dynamic_metrics = llm_response['metrics']
                prompt_eval_duration = dynamic_metrics['prompt_eval_duration']
                eval_count = dynamic_metrics['eval_count']
                eval_duration = dynamic_metrics['eval_duration']
                prompt_eval_count = dynamic_metrics['prompt_eval_count']
                total_duration = dynamic_metrics['total_duration']
                load_duration = dynamic_metrics['load_duration']
                print(f"Dynamic LLM Response: {response}")  # Debugging print statement
            else:
                # For static routes, log the predefined response based on route
                predefined_responses = best_route.responses
                # Find the closest response based on similarity to utterances
                similarities = []
                for utt in best_route.utterances:
                    utt_embedding, _ = get_embedding(utt)
                    sim = cosine_similarity(prompt_embedding, utt_embedding)
                    similarities.append(sim)
                closest_utterance_index = np.argmax(similarities)
                response = predefined_responses[closest_utterance_index]
                # Static responses have default values for dynamic metrics
                prompt_eval_duration = 0
                eval_count = 0
                eval_duration = 0
                prompt_eval_count = embed_metrics.get('prompt_eval_count', 0)
                total_duration = embed_metrics.get('total_duration', 0)
                load_duration = embed_metrics.get('load_duration', 0)
                print(f"Static Route Response: {response}")  # Debugging print statement

        # Stop resource monitoring
        is_monitoring = False
        monitor_thread.join()

        # Measure resource statistics
        process = psutil.Process()
        memory_usage_before = memory_usage_queue.queue[0] if not memory_usage_queue.empty() else process.memory_info().rss
        memory_usage_after = memory_usage_queue.queue[-1] if not memory_usage_queue.empty() else process.memory_info().rss
        avg_cpu_usage = calculate_average_cpu()
        similarity = round(similarity, 2) if similarity is not None else None

        # Network latency: time spent in network communication for the embedding request
        network_latency = total_duration - load_duration

        # Total response time: time from receiving the request to sending the response
        total_response_time = (time.time() - start_time) * 1e9  # Convert to nanoseconds

        # Calculate tokens per second for the response
        tokens_per_second = eval_count / eval_duration * 1e9 if eval_duration > 0 else 0
        tokens_per_second = round(tokens_per_second, 2)  # Round to 2 decimal points

        # Prepare log message for CSV
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message_csv = [
            timestamp, model_name, embed_model, prompt, response, route_type, route_selected,
            similarity, "cosine", str(prompt_embedding), total_duration, load_duration, prompt_eval_count,
            prompt_eval_duration, eval_count, eval_duration, tokens_per_second,
            avg_cpu_usage, memory_usage_before, memory_usage_after, memory_allocated_for_model,
            network_latency, total_response_time, route_selection_duration, llm_invoked
        ]

        # Put the log message into the CSV queue
        print(f"Logging to CSV: {log_message_csv}")  # Debugging print statement
        csv_queue.put(log_message_csv)

        # Return the response to the client
        return {
            "status": "success" if best_route else "no_match",
            "route_selected": route_name,
            "semantic_similarity_score": similarity,
            "similarity_metric": "cosine",
            "response": response
        }

    except Exception as e:
        is_monitoring = False  # Ensure monitoring stops in case of an error
        return {"status": "error", "message": str(e)}


# Function to get embeddings from the embedding model
def get_embedding(text, model=embed_model):
    response = requests.post(
        OLLAMA_API_URL,
        json={"model": model, "input": text}
    )
    response_json = response.json()
    return response_json["embeddings"][0], response_json

# Function to calculate average CPU usage during the request
def calculate_average_cpu():
    cpu_usages = []
    while not cpu_usage_queue.empty():
        cpu_usages.append(cpu_usage_queue.get())
    return round(mean(cpu_usages), 2) if cpu_usages else 0

# Function to find the best route based on the prompt
def find_best_route(prompt_embedding, routes):
    best_route = None
    best_similarity = -1

    for route in routes:
        for utterance in route.utterances:
            utterance_embedding, _ = get_embedding(utterance)
            similarity = cosine_similarity(prompt_embedding, utterance_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route

    # If best similarity is below the threshold, set best_route to None
    if best_similarity < THRESHOLD:
        best_route = None

    return best_route, best_similarity

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Main function to start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
