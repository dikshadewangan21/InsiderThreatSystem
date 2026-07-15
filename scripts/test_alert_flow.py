#!/usr/bin/env python3
"""
scripts/test_alert_flow.py

Validation script that queries the running REST API on port 9090 to test
risk scoring, alert generation, and neighborhood explainability end-to-end.
"""

import requests
import json
import sys

URL = "http://127.0.0.1:9090/predict"

def test_event(name, event_data):
    print("=" * 70)
    print(f"TESTING EVENT: {name}")
    print("=" * 70)
    print("Payload:")
    print(json.dumps(event_data, indent=2))
    print("-" * 70)
    
    try:
        r = requests.post(URL, json={"event": event_data})
        if r.status_code != 200:
            print(f"Error: Received status code {r.status_code}")
            print(r.text)
            return
            
        res = r.json()
        print(f"Risk Score   : {res['risk_score']:.6f}")
        print(f"Risk Level   : {res['risk_level']}")
        print(f"Confidence   : {res['confidence']:.2%}")
        print(f"Alert Active : {res['alert_generated']}")
        print("-" * 70)
        
        exp = res["explanation"]
        print("TOP RISK FACTORS:")
        for idx, factor in enumerate(exp["top_factors"]):
            print(f"  {idx+1}. {factor['factor']} (Weight: {factor['importance_score']:.1%})")
            print(f"     Description: {factor['description']}")
            
        print("\nTOP INFLUENTIAL NEIGHBORS:")
        for idx, neighbor in enumerate(exp["top_influential_neighbors"]):
            print(f"  {idx+1}. {neighbor['neighbor_id']} ({neighbor['node_type']}) via '{neighbor['relation']}' (Influence: {neighbor['influence_score']:.1%})")
            print(f"     Description: {neighbor['description']}")
            
        print("\nANALYST SUMMARY:")
        print(exp["analyst_summary"])
        print("\nRECOMMENDED ACTIONS:")
        print(exp["recommended_action"])
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("Error sending request to API:", e)
        print("Please verify that your Uvicorn server is running on http://127.0.0.1:9090!")

# Test logon event
logon_event = {
    "user_id": "CRD0624",
    "timestamp": 1420070400.0,
    "event_type": "logon",
    "resource": "PC-6056",
    "target_user": "none"
}

# Test HTTP domain event
http_event = {
    "user_id": "CRD0624",
    "timestamp": 1420070460.0,
    "event_type": "http",
    "resource": "msn.com",
    "target_user": "none"
}

# Test sensitive file event
file_event = {
    "user_id": "CRD0624",
    "timestamp": 1420070520.0,
    "event_type": "file",
    "resource": "doc",
    "target_user": "none"
}

if __name__ == "__main__":
    print("Starting API Flow Validation Test...\n")
    test_event("Logon Event", logon_event)
    test_event("HTTP Visit Event", http_event)
    test_event("File Event", file_event)
