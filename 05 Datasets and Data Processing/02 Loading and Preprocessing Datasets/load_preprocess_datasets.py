%% load_preprocess_datasets.py
# Setup: pip install datasets transformers matplotlib pandas
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Simulated IMDB dataset
dataset = [
    {"text": "Great movie, loved it!", "label": 1},
    {"text": "Terrible plot, boring.", "label": 0},
    {"text": "Amazing cast!", "label": 1},
    {"text": "Waste of time.", "label": 0}
]

# Function to load and preprocess dataset, and visualize results
def load_preprocess():
    print("Synthetic Data: Simulated IMDB dataset")
    print("Dataset:", [d["text"] for d in dataset])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Simulate loading and preprocessing
    processed_data = []
    for item in dataset:
        try:
            encoded = tokenizer(item["text"], truncation=True, padding="max_length", max_length=32)
            processed_data.append({"input_ids": encoded["input_ids"], "label": item["label"]})
            print(f"Processed: {item['text']}")
        except Exception as e:
            print(f"Error processing {item['text']}: {e}")
    
    # Visualize label distribution
    label_counts = Counter(d["label"] for d in dataset)
    plt.figure(figsize=(6, 4))
    plt.bar(["Negative", "Positive"], [label_counts[0], label_counts[1]], color=['red', 'green'])
    plt.title("Preprocessed Dataset Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("load_preprocess_datasets_output.png")
    print("Visualization: Preprocessed dataset saved as load_preprocess_datasets_output.png")

if __name__ == "__main__":
    load_preprocess()