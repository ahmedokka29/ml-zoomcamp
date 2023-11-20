import requests
url = 'http://127.0.0.1:9696/predict'

# customer = {
#     "job": "retired", 
#     "duration": 445, 
#     "poutcome": "success"
# }

# customer = {
#     "job": "unknown", 
#     "duration": 270, 
#     "poutcome": "failure"
#     }

customer = {
    "job": "retired", 
    "duration": 445, 
    "poutcome": "success"}


response = requests.post(url,json=customer).json()
print(response['credit_probability'])
if response['credit'] == True:
    print('Credit is approved')
else:
    print('Credit is not approved')