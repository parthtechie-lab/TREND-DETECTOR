import urllib.request
import json
import sys

url = "http://127.0.0.1:3000/api/v1/app-conversations/start-tasks/search"
try:
    with urllib.request.urlopen(url) as response:
        if response.status == 200:
            data = json.loads(response.read().decode())
            items = data.get("items", [])
            for item in items:
                if item.get("id") == sys.argv[1]:
                    print(json.dumps(item, indent=2))
                    sys.exit(0)
            print(f"Task {sys.argv[1]} not found")
        else:
            print(f"Failed to fetch. Status: {response.status}")
except Exception as e:
    print(f"Error: {e}")
