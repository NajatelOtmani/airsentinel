import requests

resp = requests.post(
    "http://localhost:8000/auth/login",
    data={"username": "admin", "password": "Test1234!"},
)
print("LOGIN:", resp.status_code, resp.text)

if resp.status_code == 200:
    token = resp.json()["access_token"]
    r = requests.get(
        "http://localhost:8000/api/v1/sensors/LONDON_BX1/latest",
        headers={"Authorization": f"Bearer {token}"},
    )
    print("SENSOR:", r.status_code, r.text)
