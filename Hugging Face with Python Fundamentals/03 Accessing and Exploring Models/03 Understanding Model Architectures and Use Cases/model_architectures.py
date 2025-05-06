# %% model_architectures.py
# Setup: pip install matplotlib pandas
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Model architectures and use cases
models = [
    {"name": "distilbert-base-uncased", "architecture": "Transformer", "use_case": "Text Classification"},
    {"name": "gpt2", "architecture": "Transformer", "use_case": "Text Generation"},
    {"name": "google/vit-base-patch16-224", "architecture": "Vision Transformer", "use_case": "Image Classification"}
]

# Function to analyze architectures and visualize results
def analyze_architectures():
    print("Synthetic Data: Model architectures")
    print("Models:", [m["name"] for m in models])
    
    # Count architectures and use cases
    arch_counts = Counter(m["architecture"] for m in models)
    use_case_counts = Counter(m["use_case"] for m in models)
    
    # Visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(arch_counts.keys(), arch_counts.values(), color='blue')
    plt.title("Model Architectures")
    plt.xlabel("Architecture")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(use_case_counts.keys(), use_case_counts.values(), color='green')
    plt.title("Model Use Cases")
    plt.xlabel("Use Case")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("model_architectures_output.png")
    print("Visualization: Architecture analysis saved as model_architectures_output.png")

if __name__ == "__main__":
    analyze_architectures()