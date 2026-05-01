import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
from model_architecture import FoodDetectionModel
import torch
from typing import Annotated
from torchvision import transforms
import PIL.Image as Image
import io
from enum import Enum
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel, Field

food_model = None
spam_model_encoder = None
spam_model = None


test_data_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

@asynccontextmanager
async def lifespan(app):
    global food_model
    global spam_model
    global spam_model_encoder

    #initializing the food model
    food_model = FoodDetectionModel(in_channels=3, hidden_units=12)
    food_model.load_state_dict(torch.load('./foodMode.pth', map_location=torch.device('cpu')))
    food_model.eval()

    #initializing the spam encoder
    spam_model_encoder = SentenceTransformer('all-MiniLM-L6-v2')

    #initializing the spam model
    spam_model = joblib.load('./spam_detection_model.joblib')

    yield
    food_model=None
    spam_model_encoder=None
    spam_model=None

app = FastAPI(lifespan=lifespan, docs_url='/')

def pre_processing(file:Annotated[UploadFile, File()]):
    image = Image.open(io.BytesIO(file.file.read())).convert('RGB')
    processed_file = test_data_transforms(image)
    return processed_file

class Classes(Enum):
    PIZZA = 0
    STEAK = 1
    SUSHI = 2


@app.post('/predict')
def predict(file:UploadFile=File(...)):
    transformed_image = pre_processing(file)
    
    with torch.inference_mode():
        logits = food_model(transformed_image.unsqueeze(dim=0))
        pred_probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_class = Classes(pred_label.item())
        pred_class = pred_class.name

    return {'FOOD NAME':pred_class}

def pre_process_spam(str:str):
    return spam_model_encoder.encode(str).reshape(1,-1)

class SpamInput(BaseModel):
    text: str=Field(...)

@app.post('/spam_detection')
def spam_detection(input_data: SpamInput):
    processed_str = pre_process_spam(input_data.text)
    prediction = spam_model.predict(processed_str)
    
    return {'is_spam': prediction[0]}
