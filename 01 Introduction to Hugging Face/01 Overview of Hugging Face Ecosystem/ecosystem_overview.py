# %% ecosystem_overview.py
# Setup: pip install transformers datasets huggingface_hub matplotlib pandas
import matplotlib.pyplot as plt
from collections import Counter

# Synthetic Data: Representing Hugging Face ecosystem components
ecosystem_components = [
    {"name": "Transformers", "purpose": "Pre-trained models for NLP, vision, audio"},
    {"name": "Datasets", "purpose": "Data loading and preprocessing"},
    {"name": "Hub", "purpose": "Model and dataset sharing"},
    {"name": "Tokenizers", "purpose": "Text tokenization for models"}
]

# Function to demonstrate and visualize ecosystem overview
def overview_ecosystem():
    print("Synthetic Data: Hugging Face ecosystem components")
    print("Components:", [comp["name"] for comp in ecosystem_components])
    
    # Count components by purpose (simplified categorization)
    purposes = [comp["purpose"].split()[0] for comp in ecosystem_components]
    purpose_counts = Counter(purposes)
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(purpose_counts.keys(), purpose_counts.values(), color='blue')
    plt.title("Hugging Face Ecosystem Components by Purpose")
    plt.xlabel("Purpose")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.savefig("ecosystem_overview_output.png")
    print("Visualization: Ecosystem overview saved as ecosystem_overview_output.png")

if __name__ == "__main__":
    overview_ecosystem()