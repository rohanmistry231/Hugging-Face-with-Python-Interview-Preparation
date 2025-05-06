# %% monitor_models.py
# Setup: pip install transformers matplotlib pandas
from transformers import pipeline
import matplotlib.pyplot as plt
import time
from collections import Counter

# Synthetic Data: Mock API requests
requests = [
    {"text": "I love this product!"},
    {"text": "This is terrible."},
    {"text": "Amazing experience!"}
]

# Function to simulate model monitoring and visualize results
def monitor_models():
    print("Synthetic Data: Mock API requests")
    print("Requests:", [r["text"] for r in requests])
    
    # Initialize classifier
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Track metrics
    response_times = []
    predictions = []
    success_counts = {"Successful": 0, "Failed": 0}
    
    for req in requests:
        try:
            start_time = time.time()
            result = classifier(req["text"])
            response_times.append(time.time() - start_time)
            predictions.append(result[0]["label"])
            success_counts["Successful"] += 1
            print(f"Text: {req['text']} -> Label: {result[0]['label']}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing {req['text']}: {e}")
    
    # Visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(response_times) + 1), response_times, marker='o', color='blue')
    plt.title("API Response Times")
    plt.xlabel("Request Number")
    plt.ylabel("Time (seconds)")
    
    plt.subplot(1, 2, 2)
    label_counts = Counter(predictions)
    plt.bar(label_counts.keys(), label_counts.values(), color='green')
    plt.title("Prediction Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("monitor_models_output.png")
    print("Visualization: Monitoring metrics saved as monitor_models_output.png")

if __name__ == "__main__":
    monitor_models()