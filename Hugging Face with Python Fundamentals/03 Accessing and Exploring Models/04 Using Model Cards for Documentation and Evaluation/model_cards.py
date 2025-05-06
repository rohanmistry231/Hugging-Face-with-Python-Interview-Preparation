# %% model_cards.py
# Setup: pip install huggingface_hub matplotlib pandas
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Simulated Model Card metadata
model_cards = [
    {"model_id": "distilbert-base-uncased", "task": "text-classification", "metrics": {"accuracy": 0.92}, "license": "Apache-2.0"},
    {"model_id": "gpt2", "task": "text-generation", "metrics": {"perplexity": 20.5}, "license": "MIT"},
    {"model_id": "google/vit-base-patch16-224", "task": "image-classification", "metrics": {"accuracy": 0.88}, "license": "Apache-2.0"}
]

# Function to analyze Model Cards and visualize results
def analyze_model_cards():
    print("Synthetic Data: Model Cards")
    print("Model Cards:", [card["model_id"] for card in model_cards])
    
    # Analyze Model Card metadata
    task_counts = Counter(card["task"] for card in model_cards)
    license_counts = Counter(card["license"] for card in model_cards)
    
    # Simulate fetching Model Card data
    success_counts = {"Successful": 0, "Failed": 0}
    for card in model_cards:
        try:
            print(f"Analyzing Model Card for {card['model_id']}: Task={card['task']}, Metrics={card['metrics']}")
            success_counts["Successful"] += 1
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error analyzing {card['model_id']}: {e}")
    
    # Visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(task_counts.keys(), task_counts.values(), color='blue')
    plt.title("Model Cards by Task")
    plt.xlabel("Task")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(license_counts.keys(), license_counts.values(), color='green')
    plt.title("Model Cards by License")
    plt.xlabel("License")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("model_cards_output.png")
    print("Visualization: Model Card analysis saved as model_cards_output.png")

if __ai__ == "__main__":
    analyze_model_cards()