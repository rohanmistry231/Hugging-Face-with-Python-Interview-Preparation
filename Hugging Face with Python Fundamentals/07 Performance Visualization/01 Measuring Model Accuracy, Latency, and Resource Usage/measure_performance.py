# %% measure_performance.py
# Setup: pip install transformers scikit-learn matplotlib pandas psutil
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score
import time
import psutil
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Mock test dataset
test_dataset = [
    {"text": "Great product!", "label": 1},
    {"text": "Awful experience.", "label": 0},
    {"text": "Fantastic service!", "label": 1},
    {"text": "Very disappointing.", "label": 0}
]

# Function to measure performance and visualize results
def measure_performance():
    print("Synthetic Data: Test dataset")
    print("Dataset:", [d["text"] for d in test_dataset])
    
    # Initialize classifier
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Measure performance
    predictions = []
    true_labels = [d["label"] for d in test_dataset]
    latencies = []
    memory_usages = []
    
    for item in test_dataset:
        try:
            start_time = time.time()
            result = classifier(item["text"])
            latencies.append(time.time() - start_time)
            predictions.append(1 if result[0]["label"] == "POSITIVE" else 0)
            memory_usages.append(psutil.Process().memory_info().rss / 1024**2)  # MB
            print(f"Text: {item['text']} -> Predicted: {predictions[-1]}")
        except Exception as e:
            print(f"Error processing {item['text']}: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    avg_latency = sum(latencies) / len(latencies)
    avg_memory = sum(memory_usages) / len(memory_usages)
    
    # Visualization
    metrics = {"Accuracy": accuracy, "F1 Score": f1, "Avg Latency (s)": avg_latency, "Avg Memory (MB)": avg_memory}
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'purple'])
    plt.title("Model Performance Metrics")
    plt.xticks(rotation=45)
    plt.savefig("measure_performance_output.png")
    print("Visualization: Performance metrics saved as measure_performance_output.png")

if __name__ == "__main__":
    measure_performance()