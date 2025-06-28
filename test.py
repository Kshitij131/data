import os
import requests
# from dotenv import load_dotenv

# load_dotenv()
# token = os.getenv("11BKLPL3Y0Luz73EQd9aDW_yAxwVyOAXa0ys8kDXk62we4tXyD2rCZqSmG2cJ7Ao7bIA2R43DGS2dh2DRB")

# headers = {
#     "Authorization": f"Bearer {token}",
#     "Accept": "application/vnd.github+json"
# }

# r = requests.get("https://api.github.com/user", headers=headers)

# print("Status Code:", r.status_code)
# print("Response:", r.json())
print("From os.environ directly:", os.environ.get("GITHUB_TOKEN"))
