import torch
import json
import hashlib
import requests

IPFS_API_URL = "http://127.0.0.1:5001/api/v0/add"

def save_model_to_ipfs(parameters):
    serialized_params = json.dumps([p.tolist() for p in parameters])
    model_hash = hashlib.sha256(serialized_params.encode()).hexdigest()
    response = requests.post(IPFS_API_URL, files={"file": serialized_params})
    if response.status_code == 200:
        cid = response.json()["Hash"]
        print(f"[INFO] Model saved on IPFS with CID: {cid}")
        return cid
    else:
        print("[ERROR] Unable to upload to IPFS")
        return None
