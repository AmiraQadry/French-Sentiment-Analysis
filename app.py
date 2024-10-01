from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define request body structure using Pydantic
class Review(BaseModel):
    review_body: str

# Create a GET endpoint for root
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Create a POST endpoint for predictions
@app.post("/predict")
async def predict(review_body: str = Form(...)):
    features = vectorizer.transform([review_body])
    prediction = model.predict(features)
    sentiment = prediction[0]  # Assuming sentiment labels are 'sad', 'neutral', 'happy'
    return {"sentiment": sentiment}

# Main entry point to run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
