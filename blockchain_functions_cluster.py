import json
from web3 import Web3

EXT_CONTRACT_ADDRESS = "0xf5c94f8BCEC7FD286901e73C64658ed0670153dA"
FED_CLUSTERING_ADDRESS = "0xc729A5D6dE82B076A0A028Be1405F3A730fAC42b"

WEB3_PROVIDER = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))

with open('./build/contracts/ExtendedHashStorage.json') as f:
    ext_contract_data = json.load(f)
EXT_ABI = ext_contract_data['abi']

def register_client(username, password, account):
    try:
        contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
        tx_hash = contract.functions.registerClient(username, password).transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        error_str = str(e)
        if "Client already registered" in error_str:
            print("Client has already registered")
            return "already_registered"
        else:
            print(f"register_client error: {e}")
            return None

def save_hash(hash_value, account):
    try:
        contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
        tx_hash = contract.functions.saveHash(hash_value).transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"save_hash error: {e}")
        return None

def check_hash_exists(hash_value, account):
    try:
        contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
        exists = contract.functions.checkIfHashExists(hash_value).call({"from": account})
        return exists
    except Exception as e:
        print(f"check_hash_exists error: {e}")
        return False

def penalize_client(client_address, amount, account):
    try:
        contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
        tx_hash = contract.functions.penalizeClient(client_address, amount).transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"penalize_client error: {e}")
        return None

def reset_tokens(account):
    try:
        contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
        tx_hash = contract.functions.resetTokenBalances().transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"reset_tokens error: {e}")
        return None

def deploy_fedclustering(dimension, num_clusters, account):
    with open("./build/contracts/FedClustering.json") as f:
        contract_data = json.load(f)
    fedclustering_contract = web3.eth.contract(abi=contract_data["abi"], bytecode=contract_data["bytecode"])
    tx_hash = fedclustering_contract.constructor(dimension, num_clusters).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    deployed_address = receipt.contractAddress
    print("FedClustering deployed at:", deployed_address)
    return deployed_address

def set_cluster_center(fedclustering_address, clusterId, newCenter, account):
    with open("./build/contracts/FedClustering.json") as f:
        contract_data = json.load(f)
    fedclustering = web3.eth.contract(address=Web3.to_checksum_address(fedclustering_address), abi=contract_data["abi"])
    tx_hash = fedclustering.functions.setClusterCenter(clusterId, newCenter).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt

def get_cluster_center(fedclustering_address, clusterId, account):
    with open("./build/contracts/FedClustering.json") as f:
        contract_data = json.load(f)
    fedclustering = web3.eth.contract(address=Web3.to_checksum_address(fedclustering_address), abi=contract_data["abi"])
    center = fedclustering.functions.getClusterCenter(clusterId).call({"from": account})
    return center

def update_cluster_center(fedclustering_address, clusterId, newCenter, account):
    with open("./build/contracts/FedClustering.json") as f:
        contract_data = json.load(f)
    fedclustering = web3.eth.contract(address=Web3.to_checksum_address(fedclustering_address), abi=contract_data["abi"])
    tx_hash = fedclustering.functions.updateClusterCenter(clusterId, newCenter).transact({"from": account})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt


