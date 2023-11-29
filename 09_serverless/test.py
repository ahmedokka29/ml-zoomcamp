import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
img_url = "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"
data = {"url":img_url}
result = requests.post(url , json=data).json()
print(result)