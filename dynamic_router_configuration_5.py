# Multi-shot, dynamic routing
# THRESHOLD value based comparison is included for "no_route"
# Multiple math functions with schema
# LLM is not called for "no_route"
# Both static and dynamic routes are selected
# 'route_selection_duration' is included in csv in ns to measure time of selecting a route 
# 'llm_invoked' is included in csv to check whether llm is invoked, 1 means yes, 0 means no, useful for dyanmic routing analysis
###########
# Configuration: 5: add function, heapsort, factorial, sentiment analysis, GCD
###########

# Install follow for sentiment analysis: pip install vaderSentiment

# Ask the code by follows
# To do add: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Add 2 with 7"}'
# To do sort: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Sort the following numbers using heap sort: [9, 4, 1, 7]"}'
# To do factorial: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Factorial of 9"}'
# To do sentiment analysis: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Analyze the sentiment of this text: I love FastAPI it is nice"}'
# To do GCD: curl -X POST http://localhost:5000/process_prompt -H "Content-Type: application/json" -d '{"prompt": "Find the GCD of 48 and 18"}'
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
import subprocess


app = FastAPI()

# Define the threshold for similarity score 
THRESHOLD = 0.5  # Adjust based on embedding model

# Define the embedding model and LLM
embed_model = "all-minilm:33m"  # Embedding model
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

def heap_sort(numbers: list) -> list:
    """Perform heap sort on a list of numbers."""
    import heapq
    heapq.heapify(numbers)
    return [heapq.heappop(numbers) for _ in range(len(numbers))]

# Add this new function to calculate factorial
def calculate_factorial(n: int) -> int:
    """Calculate the factorial of a given number."""
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)


# Define the sentiment analysis function
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define the sentiment analysis function
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of a given text."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return "positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"



# Function to calculate GCD of two numbers
def calculate_gcd(a: int, b: int) -> int:
    """
    Calculate the Greatest Common Divisor (GCD) of two numbers.
    Example:
    - Input: 48, 18
    - Output: 6
    """
    while b:
        a, b = b, a % b
    return a

        
        
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

heap_sort_schema = {
    "name": "heap_sort",
    "description": "Sort a list of numbers using heap sort.",
    "parameters": {
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The list of numbers to sort."
            }
        },
        "required": ["numbers"]
    }
}

# Add this schema for the new function
factorial_schema = {
    "name": "calculate_factorial",
    "description": "Calculate the factorial of a given number.",
    "parameters": {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "description": "The number to calculate factorial for."
            }
        },
        "required": ["n"]
    }
}

# Define the schema for the sentiment analysis
sentiment_analysis_schema = {
    "name": "analyze_sentiment",
    "description": "Analyze the sentiment of a given text.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze for sentiment."
            }
        },
        "required": ["text"]
    }
}


# GCD function schema
gcd_schema = {
    "name": "calculate_gcd",
    "description": "Calculate the Greatest Common Divisor (GCD) of two numbers.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "The first number."
            },
            "b": {
                "type": "integer",
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
        name="gcd",
        utterances=[
            "Find the GCD of 48 and 8",
            "What is the greatest common divisor of 100 and 75?",
            "Get the GCD of 54 and 24",
            "Compute the greatest common divisor of 45 and 60",
            "What is the GCD of 81 and 27?"
        ],
        dynamic=True,
        functions=[calculate_gcd],
        function_schemas=[gcd_schema]
    ),    
    Route(
        name="sentiment_analysis",
        utterances=[
            "Analyze the sentiment of this text: I love FastAPI it is nice",
            "Sentiment analysis for this sentence: I am happy for my country",
            "What is the sentiment of this text? roses are red",
            "Find the sentiment of this review: I hate losers",
            "Determine the mood of this statement: Internet is network"
        ],
        dynamic=True,
        functions=[analyze_sentiment],
        function_schemas=[sentiment_analysis_schema]
    ),    
    Route(
        name="factorial",
        utterances=[
            "Calculate the factorial of 5",
            "What is the factorial of 7",
            "Compute factorial for 9",
            "Find factorial of number 6",
            "Factorial calculation for 8"
        ],
        dynamic=True,
        functions=[calculate_factorial],
        function_schemas=[factorial_schema]
    ),
    Route(
        name="heap_sort",
        utterances=[
            "Sort the following numbers using heap sort: [9, 4, 1, 7]",
            "Heap sort these numbers [6, 4, 2, 8]",
            "Can you sort this list using heap sort? [5, 9, 1, 4]",
            "Sort numbers using heap sort [2, 4, 6, 7]",
            "Use heap sort to arrange these numbers [8, 9, 3, 7]"
        ],
        dynamic=True,
        functions=[heap_sort],
        function_schemas=[heap_sort_schema]
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
    examples = ""
    if route.name == "math_operations":
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
"""
    elif route.name == "heap_sort":
        examples = """
Example 1:
User: Sort the following numbers using heap sort: [5, 3, 8, 1]
Assistant:
Function: heap_sort
Arguments: {"numbers": [5, 3, 8, 1]}
Result: [1, 3, 5, 8]

Example 2:
User: Can you sort this list using heap sort? [10, 7, 2, 9]
Assistant:
Function: heap_sort
Arguments: {"numbers": [10, 7, 2, 9]}
Result: [2, 7, 9, 10]

Example 3:
User: Heap sort these numbers: [4, 6, 2, 5]
Assistant:
Function: heap_sort
Arguments: {"numbers": [4, 6, 2, 5]}
Result: [2, 4, 5, 6]
"""
    elif route.name == "factorial":
        examples = """
Example 1:
User: Calculate the factorial of 5
Assistant:
Function: calculate_factorial
Arguments: {"n": 5}
Result: 120

Example 2:
User: What is the factorial of 0?
Assistant:
Function: calculate_factorial
Arguments: {"n": 0}
Result: 1

Example 3:
User: Compute factorial for 3
Assistant:
Function: calculate_factorial
Arguments: {"n": 3}
Result: 6
"""
    elif route.name == "sentiment_analysis":
        examples = """
Example 1:
User: Analyze the sentiment of this text: "I am very happy with my new phone."
Assistant:
Function: analyze_sentiment
Arguments: {"text": "I am very happy with my new phone."}
Result: positive

Example 2:
User: What is the sentiment of this sentence: "The weather is terrible today."
Assistant:
Function: analyze_sentiment
Arguments: {"text": "The weather is terrible today."}
Result: negative

Example 3:
User: Find the sentiment of this statement: "I have a neutral opinion about this movie."
Assistant:
Function: analyze_sentiment
Arguments: {"text": "I have a neutral opinion about this movie."}
Result: neutral
"""
    elif route.name == "gcd":
        examples = """
Example 1:
User: Find the GCD of 48 and 18
Assistant:
Function: calculate_gcd
Arguments: {"a": 48, "b": 18}
Result: 6

Example 2:
User: Compute the GCD of 54 and 24
Assistant:
Function: calculate_gcd
Arguments: {"a": 54, "b": 24}
Result: 6

Example 3:
User: What is the greatest common divisor of 45 and 60?
Assistant:
Function: calculate_gcd
Arguments: {"a": 45, "b": 60}
Result: 15
"""
    else:
        # Handle other routes or provide default examples
        pass

    prompt_template = examples + f"""
Now, respond to this query:
User: {prompt}
Assistant:
"""
    return prompt_template


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

    # Map function names to the actual functions
    function_mapping = {
        "add": "add",
        "heap_sort": "heap_sort",
        "calculate_factorial": "calculate_factorial",
        "analyze_sentiment": "analyze_sentiment",  
        "calculate_gcd": "calculate_gcd"
    }

    for line in lines:
        if line.startswith('Function:'):
            if function_name and arguments is not None:
                function_calls.append((function_name, arguments))
            raw_function_name = line[len('Function:'):].strip()
            function_name = function_mapping.get(raw_function_name.lower(), None)
            arguments = None
        elif line.startswith('Arguments:'):
            args_text = line[len('Arguments:'):].strip()
            try:
                arguments = json.loads(args_text)
            except json.JSONDecodeError:
                arguments = None

    if function_name and arguments is not None:
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
