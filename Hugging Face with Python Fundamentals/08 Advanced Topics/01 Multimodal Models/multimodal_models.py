# %% multimodal_models.py
# Setup: pip install transformers torch matplotlib pandas
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
from collections import Counter
import time

# Synthetic Data: Mock text-image pairs
data = [
    {"text": "A cute cat", "image": "mock_cat_image", "label": "positive"},
    {"text": "A scary dog", "image": "mock_dog_image", "label": "negative"},
    {"text": "A happy bird", "image": "mock_bird_image", "label": "positive"}
]

# Function to demonstrate multimodal models and visualize results
def multimodal_models():
    print("Synthetic Data: Mock text-image pairs")
    print("Data:", [d["text"] for d in data])
    
    # Initialize CLIP model
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as e:
        print("Error initializing CLIP:", e)
        return
    
    # Simulate CLIP processing
    similarities = []
    success_counts = {"Successful": 0, "Failed": 0}
    
    for item in data:
        try:
            start_time = time.time()
            # Mock image input (simulated as text for compatibility)
            inputs = processor(text=item["text"], images=None, return_tensors="pt", padding=True)
            outputs = model.get_text_features(**inputs)
            # Simulate similarity score
            similarity = random.uniform(0.7, 0.9)  # Mock score
            similarities.append(similarity)
            success_counts["Successful"] += 1
            print(f"Text: {item['text']} -> Similarity: {similarity:.4f}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing {item['text']}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(similarities)), similarities, color='blue')
    plt.title("CLIP Text-Image Similarity Scores")
    plt.xlabel("Sample")
    plt.ylabel("Similarity Score")
    plt.savefig("multimodal_models_output.png")
    print("Visualization: CLIP results saved as multimodal_models_output.png")

if __name__ == "__main__":
    multimodal_models()