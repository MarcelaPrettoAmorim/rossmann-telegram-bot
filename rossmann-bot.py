import os
import json
import requests
import pandas as pd
from flask import Flask, request, Response

# constants
TOKEN = '6879464922:AAHoQZR3oj5IBBzpfOHnh-MPrclQmrJ71HU'

# Info about the bot
# https://api.telegram.org/bot6879464922:AAHoQZR3oj5IBBzpfOHnh-MPrclQmrJ71HU/getMe

# Get updates
# https://api.telegram.org/bot6879464922:AAHoQZR3oj5IBBzpfOHnh-MPrclQmrJ71HU/getUpdates

# Send message
# https://api.telegram.org/bot6879464922:AAHoQZR3oj5IBBzpfOHnh-MPrclQmrJ71HU/sendMessage?chat_id=6879615638&text=Hi user, I am doing fine, thanks!

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/'.format(TOKEN)
    url = url + 'sendMessage?chat_id={}'.format(chat_id)

    requests.post(url, json = {'text': text})
    print('Status Code{}'.format(r.status_code))

    return None

def load_dataset():
    # loading test dataset
    df10 = pd.read_csv( 'test.csv' )
    df_store_raw = pd.read_csv( 'store.csv' )

    # Merge test dataset + store
    df_test = pd.merge(df10, df_store_raw, how = 'left', on = 'Store')

    # Choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    if not df_test.empty:

        # Remove closed days
        df_test = df_test[df_test['Open'] != 0]
        df_test = df_test[-df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis = 1)

        # Convert Dataframe to json
        data = json.dumps(df_test.to_dict(orient = 'records'))
    else:
        data = 'error'
    
    return data

def predict(data):
    # API Call
    # url = 'http://localhost:5000/rossmann/predict' # porta 5000 é a porta padrão do Flask
    url = 'https://api-rossmann-sales-predict.onrender.com/rossmann/predict'
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data = data, headers = header)
    print('Status Code {}'.format(r.status_code))

    d1 = pd.DataFrame(r.json(), columns = r.json()[0].keys())

    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    
    except ValueError:
        send_message(chat_id, 'Store ID is wrong.')
        store_id = 'error'     

    return chat_id, store_id


# API initialize
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])

def index():
    if request.method == 'POST':
       message = request.get_json()

       chat_id, store_id = parse_message(message)

       if store_id != 'error':
           # loading data
           data = load_dataset(store_id)
        
           if data != 'error':
            # prediction
            d1 = predict(data)
            # calculation
            d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()

            # send message
            msg = ('Store number {} will sell R${:,.2f} in the next 6 weeks'.format(d2['store'].values[0], d2['prediction'].values[0]))

            send_message(chat_id, msg)
            return Response('OK', status = 200)
           
           else:
            send_message(chat_id, "Store not available.")
            return Response('OK', status = 200)

       else:
           send_message(chat_id, 'Store ID is wrong.')
           return Response ('OK', status = 200)

    else:
        return '<h1>Rossmann TelegramBOT</h1>'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port)

