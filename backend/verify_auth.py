import requests
import json

BASE_URL = "http://localhost:8000"

def test_registration_valid():
    print("\nTesting valid registration...")
    payload = {
        "email": "testuser@example.com",
        "password": "Password123!"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_registration_invalid_password():
    print("\nTesting invalid password (no uppercase)...")
    payload = {
        "email": "invalid@example.com",
        "password": "password123!"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_registration_duplicate_email():
    print("\nTesting duplicate email...")
    payload = {
        "email": "testuser@example.com",
        "password": "Password123!"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_login_valid():
    print("\nTesting valid login...")
    payload = {
        "email": "testuser@example.com",
        "password": "Password123!"
    }
    response = requests.post(f"{BASE_URL}/auth/login", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_login_invalid_credentials():
    print("\nTesting invalid login...")
    payload = {
        "email": "testuser@example.com",
        "password": "WrongPassword!"
    }
    response = requests.post(f"{BASE_URL}/auth/login", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    try:
        test_registration_valid()
        test_registration_invalid_password()
        test_registration_duplicate_email()
        test_login_valid()
        test_login_invalid_credentials()
    except Exception as e:
        print(f"Error during testing: {e}")
