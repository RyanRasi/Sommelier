# test_api.py
import requests

BASE_URL = "http://localhost:8000"

# Test 1: Root endpoint
r = requests.get(f"{BASE_URL}/")
assert r.status_code == 200
assert r.json()['status'] == 'running'
print("✅ Root endpoint OK")

# Test 2: Health check
r = requests.get(f"{BASE_URL}/health")
assert r.status_code == 200
data = r.json()
assert data['status'] == 'healthy'
assert data['wines_loaded'] > 0
assert 'model' in data
assert 'version' in data
print(f"✅ Health check OK — {data['wines_loaded']} wines loaded")

# Test 3: Valid recommendation
r = requests.post(f"{BASE_URL}/recommend",
                  json={"query": "fruity wine"})
assert r.status_code == 200
data = r.json()
assert data['count'] > 0
assert 'elapsed_seconds' in data
print("✅ Recommend endpoint OK")

# Test 4: Response shape correct
rec = data['recommendations'][0]
for field in ['rank', 'title', 'why', 'food_pairing', 'serving_tip']:
    assert field in rec, f"Missing field: {field}"
print("✅ Response shape correct")

# Test 5: Empty query returns structured error
r = requests.post(f"{BASE_URL}/recommend",
                  json={"query": ""})
assert r.status_code == 400
error = r.json()
assert 'error' in error
assert 'status_code' in error
print("✅ Empty query returns structured error")

# Test 6: Oversized query rejected
r = requests.post(f"{BASE_URL}/recommend",
                  json={"query": "wine " * 200})
assert r.status_code == 400
print("✅ Oversized query correctly rejected")

# Test 7: CORS headers present
r = requests.options(
    f"{BASE_URL}/recommend",
    headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
)
assert "access-control-allow-origin" in r.headers
print("✅ CORS headers present")

# Test 8: Unknown origin blocked
r = requests.options(
    f"{BASE_URL}/recommend",
    headers={"Origin": "http://malicious-site.com",
             "Access-Control-Request-Method": "POST"}
)
assert r.headers.get("access-control-allow-origin", "") != \
       "http://malicious-site.com"
print("✅ Unknown origin blocked")

print("\n✅ All tests passed.")