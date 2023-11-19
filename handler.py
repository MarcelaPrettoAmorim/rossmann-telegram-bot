import pickle
import json
import pandas as pd

from rossmann.Rossmann import Rossmann
from flask import Flask, request, Response

# loading model
model = pickle.load(open('/home/marcela-pretto-amorim/ds/repos/curso-ds-producao/model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask( __name__ )
@app.route('/rossmann/predict', methods=['POST'])

def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance(test_json, dict): # Unique example
            test_raw = pd.DataFrame(test_json, index = [0])
        else: # Multiple examples
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys())
        
        # Instantiate Rossmann class
        pipeline = Rossmann()
        
        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # Feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # Data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
    else:
        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    app.run('localhost') # parâmetro 0.0.0.0 indica que irá rodar na máquina local