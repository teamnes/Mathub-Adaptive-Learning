from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
 
# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
 
app = FastAPI()
 
# Load the model and tokenizer
model_path = "bert-base-uncased"  # Use your local model path if needed
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
 
class Item(BaseModel):
    answer: str
 
@app.post("/score/")
async def score_answer(item: Item):
    answer = item.answer
    if preprocess(answer).strip() == '':  # Ensure non-empty, non-stopword input
        return {"score": 0.0}
    input_ids = tokenizer.encode(preprocess(answer), return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        logits = model(input_ids)[0]
        probabilities = F.softmax(logits, dim=1)
        score = probabilities[:, 1].item()  # Assuming 1 is the index of the positive class
    return {"score": score}
 
def preprocess(text):
    # Example preprocessing function that removes stopwords
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.lower().split() if word not in stop_words])
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
