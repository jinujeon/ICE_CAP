import json
import urllib.request

url = "http://127.0.0.1:8000/home/change_stat"  # URL

d = {'cam_id': 0, 'cam_status': 'warning', 'cam_location': '공학관1층 복도', 'trash': False, 'instrusion': False, 'fallen': False}
params = json.dumps(d).encode("utf-8")  
req = urllib.request.Request(url, data=params,
                             headers={'content-type': 'application/json'})
response = urllib.request.urlopen(req)
print(response.read().decode('utf8'))  
