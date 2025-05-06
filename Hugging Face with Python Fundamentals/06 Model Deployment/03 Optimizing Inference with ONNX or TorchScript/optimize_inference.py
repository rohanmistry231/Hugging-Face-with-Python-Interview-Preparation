# %% optimize_inference.py
# Setup: pip install transformers onnx torch matplotlib pandas
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
import matplotlib.pyplot as plt

# Synthetic Data: Mock inference requests
requests = [
    {"text": "I love this product!"},
    {"text": "This is terrible."},
    {"text": "Amazing experience!"}
]

# Function to simulate ONNX optimization and visualize performance
def optimize_inference():
    print("Synthetic Data: Mock inference requests")
    print("Requests:", [r["text"] for r in requests])
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Track inference times
    standard_times = []
    optimized_times = []
    
    for req in requests:
        try:
            # Standard inference
            start_time = time.time()
            inputs = tokenizer(req["text"], return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            standard_times.append(time.time() - start_time)
            
            # Simulate ONNX-optimized inference (mock 20% faster)
            optimized_times.append(0.8 * (time.time() - start_time))
            print(f"Processed: {req['text']}")
        except Exception as e:
            print(f"Error processing {req['text']}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(standard_times) + 1), standard_times, label="Standard", marker='o', color='blue')
    plt.plot(range(1, len(optimized_times) + 1), optimized_times, label="ONNX", marker='o', color='green')
    plt.title("Inference Time Comparison")
    plt.xlabel("Request Number")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("optimize_inference_output.png")
    print("Visualization: Inference performance saved as optimize_inference_output.png")

if __name__ == "__main__":
    optimize_inference()