from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from preprocessing import preprocess_text

app = FastAPI(title="Automatic Ticket Classification API", version="1.0.0")

# Load model and mapping
model_path = "best_model_pipeline.pkl"
mapping_path = "topic_mapping.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(mapping_path):
    raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(mapping_path, 'rb') as f:
    topic_mapping = pickle.load(f)

class TicketRequest(BaseModel):
    complaint_text: str

class TicketResponse(BaseModel):
    predicted_topic: str
    topic_id: int
    complaint_text: str

@app.get("/")
def read_root():
    return {
        "message": "Automatic Ticket Classification API",
        "endpoints": {
            "/predict": "POST - Classify a ticket complaint",
            "/health": "GET - Health check",
            "/topics": "GET - Get all topic categories"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/topics")
def get_topics():
    return {"topics": topic_mapping}

@app.post("/predict", response_model=TicketResponse)
def predict_ticket(ticket: TicketRequest):
    try:
        # Preprocess the text
        processed_text = preprocess_text(ticket.complaint_text)
        
        # Make prediction
        prediction = model.predict([processed_text])[0]
        
        # Get topic name from mapping
        topic_name = topic_mapping.get(prediction, "Unknown")
        
        return TicketResponse(
            predicted_topic=topic_name,
            topic_id=int(prediction),
            complaint_text=ticket.complaint_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

