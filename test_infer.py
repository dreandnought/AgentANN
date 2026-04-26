import urllib.request
import json

try:
    print("Sending request to /api/infer...")
    req = urllib.request.Request(
        "http://localhost:8000/api/infer",
        data=json.dumps({"features": [5.1, 3.5, 1.4, 0.2]}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Response:", response.read().decode("utf-8"))
except Exception as e:
    print("Error:", e)
