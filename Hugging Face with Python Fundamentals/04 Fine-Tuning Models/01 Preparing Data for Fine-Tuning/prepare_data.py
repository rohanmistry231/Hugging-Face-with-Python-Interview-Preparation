# %% prepare_data.py
# Setup: pip install transformers datasets matplotlib pandas
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# Synthetic Data: Simulated text classification dataset
dataset = [
    {"text": "I love this product!", "label": 1},
    {"text": "This is terrible.", "label": 0},
    {"text": "Amazing experience!", "label": 1},
    {"text": "Not good at all.", "label": 0}
]

# Function to prepare data and visualize statistics
def prepare_data():
    print("Synthetic Data: Text classification dataset")
    print("Dataset:", [d["text"] for d in dataset])
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Preprocess data
    encodings = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in dataset:
        try:
            encoded = tokenizer(item["text"], truncation=True, padding="max_length", max_length=32)
            encodings["input_ids"].append(encoded["input_ids"])
            encodings["attention_mask"].append(encoded["attention_mask"])
            encodings["labels"].append(item["label"])
            print(f"Processed: {item['text']}")
        except Exception as e:
            print(f"Error processing {item['text']}: {e}")
    
    # Visualize dataset statistics
    label_counts = Counter(d["label"] for d in dataset)
    plt.figure(figsize=(6, 4))
    plt.bar(["Negative", "Positive"], [label_counts[0], label_counts[1]], color=['red', 'green'])
    plt.title("Dataset Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("prepare_data_output.png")
    print("Visualization: Dataset statistics saved as prepare_data_output.png")

if __name__ == "__main__":
    prepare_data()