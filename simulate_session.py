import requests
import pandas as pd

# Set the IP address and port of the server
IP_ADDRESS = "192.168.1.100"
PORT = 8000

doc_dir = "./Data/newsgroup_sub_500.json"
df = pd.read_json(doc_dir)
documents = df.text.values.tolist()
# Set the endpoint URL for the POST request
# url = f"http://{IP_ADDRESS}:{PORT}/api/data"
# url = 'http://127.0.0.1:5010/recommend_document'
url = 'localhost:1990/recommend_document'

# Define the data to send in the POST request
data = {  
   "label": "s",
   "doc_id": 221,
   "response_time": 3
}



doc_count = 0
while doc_count < len(df):
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("POST request successful!")
    else:
        print("POST request failed.")
        print(response.json())

    received_data = response.json()
    doc_id = received_data['document_id']
    random_document = received_data['document_id']
    print('-'*10)
    print('New Document {}'.format(random_document))
    print('-'*10)
    print('model prediction is : {}'.format(received_data['prediction']))

    print('----------')
    val = input("Please enter your decided label below: ")
    print('----------')

    data = {'label': str(val), 'doc_id': doc_id, 'response_time': 3}

    doc_count += 1
