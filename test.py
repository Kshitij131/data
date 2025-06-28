import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json"
}

response = requests.get("https://api.github.com/rate_limit", headers=headers)
data = response.json()

core = data["rate"]
print("🔋 Core Rate Limit Status:")
print(f"  Limit: {core['limit']}")
print(f"  Remaining: {core['remaining']}")
print(f"  Resets At: {core['reset']}")
