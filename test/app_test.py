import requests

url = "http://127.0.0.1:5000/"
img = {'image': open("docs/test/screenshot.png", 'rb')}
req = requests.post(url, files=img)

print(req.text)