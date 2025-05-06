# %% explore_hub.py
# Setup: pip install huggingface_hub requests matplotlib pandas
from huggingface_hub import HfApi
import requests
import matplotlib.pyplot as plt
from collections import Counter

# Function to explore Hugging Face Hub and visualize results
def explore_hub():
    print("Exploring Hugging Face Hub...")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Synthetic Data: Simulate querying models for specific tasks
    tasks = ["text-classification", "question-answering", "image-classification"]
    model_counts = Counter()
    success_counts = {"Successful": 0, "Failed": 0}
    
    for task in tasks:
        try:
            # Query models for the task (limited to 5 for simplicity)
            models = api.list_models(filter=task, sort="downloads", direction=-1, limit=5)
            model_counts[task] = len(models)
            success_counts["Successful"] += 1
            print(f"Task {task}: Found {len(models)} models")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error querying task {task}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(model_counts.keys(), model_counts.values(), color='blue')
    plt.title("Models per Task in Hugging Face Hub")
    plt.xlabel("Task")
    plt.ylabel("Model Count")
    plt.xticks(rotation=45)
    plt.savefig("explore_hub_output.png")
    print("Visualization: Hub exploration saved as explore_hub_output.png")

if __name__ == "__main__":
    explore_hub()