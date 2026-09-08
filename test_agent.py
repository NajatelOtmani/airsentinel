import time

import requests

BASE_URL = "http://localhost:8000"

try:
    login_resp = requests.post(
        f"{BASE_URL}/auth/login", data={"username": "admin", "password": "Test1234!"}
    )
    login_resp.raise_for_status()
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    print("Sending agent query for LONDON_MY7...")
    start = time.time()
    r = requests.post(
        f"{BASE_URL}/api/v1/agent/",
        json={"query": "What's the current air quality for LONDON_MY7?"},
        headers=headers,
        timeout=120,
    )
    elapsed = time.time() - start

    print(f"\nStatus: {r.status_code} | Elapsed: {elapsed:.1f}s")
    print("Response:\n", r.text[:1500])

except Exception as e:
    print(f"Error: {e}")
