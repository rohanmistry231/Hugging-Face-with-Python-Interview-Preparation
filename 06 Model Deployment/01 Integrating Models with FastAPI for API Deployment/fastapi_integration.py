%% fastapi_integration.py
# Setup: pip install fastapi uvicorn transformers matplotlib pandas
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import matplotlib.pyplot as plt
import time
from collections import Counter

# FastAPI app
app = FastAPI()
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Request model
class TextInput(BaseModel):
    text: str

# Synthetic Data: Mock API requests
requests = [
    {"text": "I love this product!"},
    {"text": "This is terrible."},
    {"text": "Amazing experience!"}
]

# Function to simulate API calls and visualize response times
def simulate_api_calls():
    print("Synthetic Data: Mock API requests")
    print("Requests:", [r["text"] for r in requests])
    
    # Track response times
    response_times = []
    success_counts = {"Successful": 0, "Failed": 0}
    
    for req in requests:
        try:
            start_time = time.time()
            result = classifier(req["text"])
            response_times.append(time.time() - start_time)
            success_counts["Successful"] += 1
            print(f"Text: {req['text']} -> Label: {result[0]['label']}")
        except Exception as e:
            success_counts["Failed"] += 1
            print(f"Error processing {req['text']}: {e}")
    
    # Visualization
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(response_times) + 1), response_times, marker='o', color='blue')
    plt.title("API Response Times")
    plt.xlabel("Request Number")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("fastapi_integration_output.png")
    print("Visualization: API response times saved as fastapi_integration_output.png")

# FastAPI endpoint
@app.post("/predict")
async def predict(input: TextInput):
    try:
        result = classifier(input.text)
        return {"prediction": result[0]["label"], "score": result[0]["score"]}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    simulate_api_calls()
    # Run with: uvicorn fastapi_integration:app --reload