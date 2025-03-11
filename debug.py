import requests
import json

# Configurazione dell'API di IPFS (modifica se necessario)
IPFS_API_URL = "http://127.0.0.1:5001/api/v0"


def add_file_to_ipfs(file_content):
    """
    Aggiunge un file (il contenuto passato come stringa) a IPFS e restituisce il CID.
    """
    files = {
        "file": ("debug.txt", file_content)
    }
    response = requests.post(f"{IPFS_API_URL}/add", files=files)
    if response.status_code == 200:
        result = response.json()
        cid = result.get("Hash")
        print(f"[INFO] File aggiunto a IPFS con CID: {cid}")
        return cid
    else:
        print(f"[ERROR] Impossibile aggiungere il file a IPFS. Status code: {response.status_code}")
        print(response.text)
        return None


def cat_file_get(cid):
    """
    Tenta di recuperare il contenuto associato al CID usando il metodo GET.
    """
    url = f"{IPFS_API_URL}/cat?arg={cid}"
    print("\n[DEBUG] Eseguo GET su /cat...")
    response = requests.get(url)
    print(f"[DEBUG] GET status: {response.status_code}")
    if response.status_code == 200:
        print("[INFO] Contenuto recuperato (GET):")
        print(response.text)
    else:
        print(f"[ERROR] GET fallita. Risposta: {response.text}")


def cat_file_post(cid):
    """
    Tenta di recuperare il contenuto associato al CID usando il metodo POST.
    """
    url = f"{IPFS_API_URL}/cat?arg={cid}"
    print("\n[DEBUG] Eseguo POST su /cat...")
    response = requests.post(url)
    print(f"[DEBUG] POST status: {response.status_code}")
    if response.status_code == 200:
        print("[INFO] Contenuto recuperato (POST):")
        print(response.text)
    else:
        print(f"[ERROR] POST fallita. Risposta: {response.text}")


def main():
    # Contenuto di test per il file
    file_content = "Questo è un file di debug per IPFS.\nContenuto di esempio per testare l'endpoint cat."

    # Aggiungi il file a IPFS e ottieni il CID
    cid = add_file_to_ipfs(file_content)
    if not cid:
        print("[ERROR] Impossibile procedere senza un CID valido.")
        return

    # Testa l'endpoint /cat usando GET e POST
    cat_file_get(cid)
    cat_file_post(cid)


if __name__ == "__main__":
    main()
