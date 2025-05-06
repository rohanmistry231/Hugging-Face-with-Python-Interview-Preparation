# %% explore_datasets_library.py
# Setup: pip install datasets matplotlib pandas
from datasets import list_datasets
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Simulated dataset metadata
dataset_metadata = [
    {"name": "imdb", "category": "text", "size": 1000},
    {"name": "squad", "category": "text", "size": 500},
    {"name": "coco", "category": "image", "size": 2000}
]

# Function to explore Datasets library and visualize results
def explore_datasets():
    print("Synthetic Data: Dataset metadata")
    print("Datasets:", [d["name"] for d in dataset_metadata])
    
    # Simulate querying available datasets
    try:
        datasets = list_datasets()[:10]  # Limit for efficiency
        print(f"Found {len(datasets)} datasets in Hugging Face Hub")
    except Exception as e:
        print("Error querying datasets:", e)
    
    # Analyze synthetic metadata
    category_counts = Counter(d["category"] for d in dataset_metadata)
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(category_counts.keys(), category_counts.values(), color='blue')
    plt.title("Datasets by Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.savefig("explore_datasets_library_output.png")
    print("Visualization: Dataset exploration saved as explore_datasets_library_output.png")

if __name__ == "__main__":
    explore_datasets()