import requests
import numpy as np
import json
import time

# Wait for server to start
time.sleep(2)

url = "http://localhost:5000/api/analyze"
headers = {"Content-Type": "application/json"}

# Generate random observation (24 features)
observation = np.random.randn(24).tolist()
payload = {"observation": observation}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("SUCCESS: API returned 200 OK")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"FAILED: API returned {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"ERROR: Could not connect to API: {e}")
