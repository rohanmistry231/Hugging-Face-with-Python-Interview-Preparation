%% create_custom_datasets.py
# Setup: pip install datasets matplotlib pandas
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Custom text classification dataset
custom_data = [
    {"text": "I love this app!", "label": 1},
    {"text": "Horrible user experience.", "label": 0},
    {"text": "Really intuitive design!", "label": 1},
    {"text": "Crashes all the time.", "label": 0}
]

# Function to create custom dataset and visualize results
def create_custom_dataset():
    print("Synthetic Data: Custom text classification dataset")
    print("Dataset:", [d["text"] for d in custom_data])
    
    # Create Hugging Face Dataset
    try:
        dataset = Dataset.from_pandas(pd.DataFrame(custom_data))
        print("Custom dataset created:", dataset)
    except Exception as e:
        print("Error creating dataset:", e)
        return
    
    # Visualize dataset statistics
    label_counts = Counter(d["label"] for d in custom_data)
    plt.figure(figsize=(6, 4))
    plt.bar(["Negative", "Positive"], [label_counts[0], label_counts[1]], color=['red', 'green'])
    plt.title("Custom Dataset Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("create_custom_datasets_output.png")
    print("Visualization: Custom dataset saved as create_custom_datasets_output.png")

if __name__ == "__main__":
    create_custom_dataset()