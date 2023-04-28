import requests

# Set the IP address and port of the server
IP_ADDRESS = "192.168.1.100"
PORT = 8000

# Set the endpoint URL for the POST request
# url = f"http://{IP_ADDRESS}:{PORT}/api/data"
url = 'http://127.0.0.1:5010/recommend_document'

# Define the data to send in the POST request
data = {  
   "label": "s",
   "doc_id": 221,
   "response_time": 3
}


# Make the POST request to the server
response = requests.post(url, json=data)

print(response)
# Check the response from the server
if response.status_code == 200:
    print("POST request successful!")
else:
    print("POST request failed.")