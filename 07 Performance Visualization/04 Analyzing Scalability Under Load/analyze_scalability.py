%% analyze_scalability.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
import time
import random

# Synthetic Data: Mock test dataset
test_dataset = [
    {"text": "Great product!"},
    {"text": "Awful experience."},
    {"text": "Fantastic service!"},
    {"text": "Very disappointing."}
]

# Function to analyze scalability and visualize results
def analyze_scalability():
    print("Synthetic Data: Test dataset")
    print("Dataset:", [d["text"] for d in test_dataset])
    
    # Initialize classifier
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Simulate load levels
    load_levels = [10, 50, 100]  # Number of requests
    avg_latencies = []
    
    for load in load_levels:
        latencies = []
        for _ in range(load):
            try:
                start_time = time.time()
                text = random.choice(test_dataset)["text"]
                classifier(text)
                latencies.append(time.time() - start_time)
            except Exception as e:
                print(f"Error processing request: {e}")
        avg_latencies.append(sum(latencies) / len(latencies))
        print(f"Load {load} requests: Avg Latency {avg_latencies[-1]:.4f} seconds")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.plot(load_levels, avg_latencies, marker='o', color='blue')
    plt.title("Scalability Under Load")
    plt.xlabel("Number of Requests")
    plt.ylabel("Average Latency (seconds)")
    plt.grid(True)
    plt.savefig("analyze_scalability_output.png")
    print("Visualization: Scalability analysis saved as analyze_scalability_output.png")

if __name__ == "__main__":
    analyze_scalability()