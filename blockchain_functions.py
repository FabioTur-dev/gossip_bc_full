import json
from web3 import Web3

WEB3_PROVIDER = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FE9bbb1938fB788471a99DD14336C3Bde51A57a"

with open('./build/contracts/ExtendedHashStorage.json') as f:
    contract_data = json.load(f)
    ABI = contract_data['abi']

web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
contract = web3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=ABI)

def register_client(username, password, account):
    try:
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
        tx_hash = contract.functions.saveHash(hash_value).transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"save_hash error: {e}")
        return None

def check_hash_exists(hash_value, account):
    try:
        exists = contract.functions.checkIfHashExists(hash_value).call({"from": account})
        return exists
    except Exception as e:
        print(f"check_hash_exists error: {e}")
        return False

def penalize_client(client_address, amount, account):
    try:
        tx_hash = contract.functions.penalizeClient(client_address, amount).transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"penalize_client error: {e}")
        return None

def reset_tokens(account):
    try:
        tx_hash = contract.functions.resetTokenBalances().transact({"from": account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt
    except Exception as e:
        print(f"reset_tokens error: {e}")
        return None


