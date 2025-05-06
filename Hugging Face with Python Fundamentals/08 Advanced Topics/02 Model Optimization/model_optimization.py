# %% model_optimization.py
# Setup: pip install transformers torch optimum matplotlib pandas
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import time

# Synthetic Data: Mock inference requests
requests = [
    {"text": "I love this product!"},
    {"text": "This is terrible."},
    {"text": "Amazing experience!"}
]

# Function to simulate model optimization and visualize results
def model_optimization():
    print("Synthetic Data: Mock inference requests")
    print("Requests:", [r["text"] for r in requests])
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Measure standard and optimized performance
    standard_times = []
    quantized_times = []
    
    for req in requests:
        try:
            # Standard inference
            start_time = time.time()
            inputs = tokenizer(req["text"], return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            standard_times.append(time.time() - start_time)
            
            # Simulate quantized model (mock 30% faster)
            quantized_times.append(0.7 * standard_times[-1])
            print(f"Processed: {req['text']}")
        except Exception as e:
            print(f"Error processing {req['text']}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(standard_times) + 1), standard_times, label="Standard", marker='o', color='blue')
    plt.plot(range(1, len(quantized_times) + 1), quantized_times, label="Quantized", marker='o', color='green')
    plt.title("Inference Time: Standard vs. Quantized")
    plt.xlabel("Request Number")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_optimization_output.png")
    print("Visualization: Optimization results saved as model_optimization_output.png")

if __name__ == "__main__":
    model_optimization()