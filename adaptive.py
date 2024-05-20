import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

with open('random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)
#print(random_forest_model.predict([["0.75","0.5","0.23","0.45","0.83"]]))
# Create FastAPI app instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (update this with your specific requirements)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
 
# Define request body model
class InputData(BaseModel):
    Accuracy: float
    Speed: int
    Confidence: float
    Critical_Thinking: float
    Subject_Mastery: float
 
# Define prediction routes

@app.post("/predict/random_forest/")
async def predict_random_forest(data: InputData):     
    input_data=[[data.Accuracy, data.Speed, data.Confidence, data.Critical_Thinking, data.Subject_Mastery]]
    prediction = random_forest_model.predict(input_data)
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)  # Specify the port here