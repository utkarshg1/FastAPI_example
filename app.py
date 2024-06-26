from DataValidation import IrisData
import uvicorn
from fastapi import FastAPI
import pandas as pd
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# Initialize Fast api app
app = FastAPI()
# Load preprocessor
with open('notebook/pre.pkl','rb') as file1:
    pre = pickle.load(file1)
# Load Model
with open('notebook/model.pkl', 'rb') as file2:
    model = pickle.load(file2)

@app.get('/')
def index():
    return {"message":"Welcome to Iris Species predictor"}

@app.post('/predict')
def predict_species(data:IrisData):
    logging.info(f'Data loaded : data')
    data = data.dict()
    sep_len = data['sepal_length']
    sep_wid = data['sepal_width']
    pet_len = data['petal_length']
    pet_wid = data['petal_width']
    xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
    logging.info(xnew)
    xnew.columns = pre.get_feature_names_out()
    xnew_pre = pre.transform(xnew)
    logging.info(xnew_pre)
    pred = model.predict(xnew_pre)
    probs = model.predict_proba(xnew_pre)
    logging.info(f'Predictions : {pred}, Probabilities : {probs}')
    return {
        'species':pred[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
