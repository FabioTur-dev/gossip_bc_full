import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST, STL10, ImageNet
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import random
import threading
from web3 import Web3
from ipfs_utils import save_model_to_ipfs
from blockchain_functions import register_client, save_hash, check_hash_exists, penalize_client, reset_tokens
import plots
from nets import NetMNIST, NetCIFAR, NetSTL, NetImageNet

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEB3_PROVIDER = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "0x5FE9bbb1938fB788471a99DD14336C3Bde51A57a"
with open("./build/contracts/ExtendedHashStorage.json") as f:
    contract_data = json.load(f)
    ABI = contract_data["abi"]

web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
contract = web3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=ABI)
accounts = web3.eth.accounts

def load_data_mnist(n_peers):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = MNIST("./data", train=False, download=True, transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def load_data_cifar10(n_peers):
    transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = CIFAR10("./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10("./data", train=False, download=True, transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def load_data_cifar100(n_peers):
    transform = Compose([ToTensor(), Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    dataset = CIFAR100("./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR100("./data", train=False, download=True, transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def load_data_fmnist(n_peers):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = FashionMNIST("./data", train=True, download=True, transform=transform)
    test_dataset = FashionMNIST("./data", train=False, download=True, transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def load_data_stl10(n_peers):
    transform = Compose([ToTensor(), Normalize((0.4467, 0.4398, 0.4066), (0.224, 0.221, 0.223))])
    dataset = STL10("./data", split="train", download=True, transform=transform)
    test_dataset = STL10("./data", split="test", download=True, transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def load_data_imagenet(n_peers):
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    dataset = ImageNet("./data/imagenet", split="train", transform=transform)
    test_dataset = ImageNet("./data/imagenet", split="val", transform=transform)
    partition_size = len(dataset) // n_peers
    lengths = [partition_size] * n_peers
    datasets = random_split(dataset, lengths)
    return datasets, test_dataset

def serialize_model_params(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]

def deserialize_model_params(model, params):
    for i, p in enumerate(model.parameters()):
        p.data = torch.from_numpy(params[i]).to(device)

def aggregate_models(model_params_list):
    if not model_params_list:
        return None
    aggregated_params = [np.zeros_like(param) for param in model_params_list[0]]
    for params in model_params_list:
        for i, param in enumerate(params):
            aggregated_params[i] += param / len(model_params_list)
    return aggregated_params

def create_topology(num_peers, num_segments):
    segments_list = []
    group_size = num_peers // num_segments
    remainder = num_peers % num_segments
    start = 0
    for i in range(num_segments):
        extra = 1 if i < remainder else 0
        segment = list(range(start, start + group_size + extra))
        segments_list.append(segment)
        start += group_size + extra
    topology = {}
    for segment in segments_list:
        for peer in segment:
            topology[peer] = [p for p in segment if p != peer]
    print("Segmenti:")
    for idx, seg in enumerate(segments_list):
        print(f"Segmento {idx}: {seg}")
    return topology

class Peer:
    def __init__(self, peer_id, dataset, test_dataset, topology, peers_dict, account, total_rounds, round_barrier, NetClass, net_params):
        self.peer_id = peer_id
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.model = NetClass(**net_params).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.neighbors = topology[peer_id]
        self.peers = peers_dict
        self.round = 0
        self.total_rounds = total_rounds
        self.round_barrier = round_barrier
        self.stop_flag = False
        self.account = account
        self.thread = threading.Thread(target=self.run)
        self.accuracy_history = []
        self.loss_history = []
        # Parametri per DP (puoi regolarli se necessario)
        self.dp_clip_threshold = 100.0
        self.dp_noise_multiplier = 0

    def train_one_epoch(self):
        print(f"Peer {self.peer_id} - Inizio training (Round {self.round + 1})...")
        self.model.train()
        trainloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        total_loss = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = self.dp_clip_threshold / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * self.dp_noise_multiplier * self.dp_clip_threshold
                    p.grad.data.add_(noise)
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(trainloader)
        self.loss_history.append(avg_loss)
        print(f"Peer {self.peer_id} - Training loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, record=False):
        print(f"Peer {self.peer_id} - Inizio testing (Round {self.round + 1})...")
        self.model.eval()
        testloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Peer {self.peer_id} - Test Accuracy: {accuracy:.2f}%")
        if record:
            self.accuracy_history.append(accuracy)
        return accuracy

    def save_cid_to_blockchain(self, model_cid):
        receipt = save_hash(model_cid, self.account)
        if receipt:
            print(f"📜 [BLOCKCHAIN] Peer {self.peer_id} - CID salvato: {model_cid}")
        else:
            print(f"Peer {self.peer_id} - Errore nel salvataggio del CID su blockchain")

    def check_cid(self, model_cid):
        exists = check_hash_exists(model_cid, self.account)
        if exists:
            print(f"✅ [BLOCKCHAIN] Peer {self.peer_id} - CID {model_cid} esiste")
        else:
            print(f"😒 [BLOCKCHAIN] Peer {self.peer_id} - CID {model_cid} NON esiste")

    def share_model(self):
        print(f"Peer {self.peer_id} - Condivisione del modello (Round {self.round + 1})...")
        params = serialize_model_params(self.model)
        model_cid = save_model_to_ipfs(params)
        print(f"Peer {self.peer_id} - Modello salvato su IPFS: {model_cid}")
        self.save_cid_to_blockchain(model_cid)
        for neighbor_id in self.neighbors:
            print(f"Peer {self.peer_id} - Invio a Peer {neighbor_id}")
            self.send_model_to_peer(neighbor_id, params, model_cid)

    def send_model_to_peer(self, neighbor_id, params, model_cid):
        if neighbor_id in self.peers:
            self.peers[neighbor_id].receive_model(self.peer_id, params, model_cid)
        else:
            print(f"Peer {self.peer_id} - Errore: Peer {neighbor_id} non trovato")

    def receive_model(self, sender_id, params, model_cid):
        print(f"Peer {self.peer_id} - Ricevuto modello da Peer {sender_id} con CID: {model_cid}")
        self.check_cid(model_cid)
        temp_model = self.model.__class__(**self.model.__dict__['_parameters'])  # Un modo per ricreare il modello (o potresti istanziarlo nuovamente con gli stessi parametri)
        deserialize_model_params(temp_model, params)
        temp_model.to(device)
        temp_model.eval()
        testloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = temp_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        received_accuracy = 100 * correct / total
        print(f"Peer {self.peer_id} - Accuracy del modello ricevuto: {received_accuracy:.2f}%")
        current_accuracy = self.test_model()
        print(f"Peer {self.peer_id} - Accuracy del modello attuale: {current_accuracy:.2f}%")
        self.aggregate_received_model(params)

    def aggregate_received_model(self, received_params):
        print(f"Peer {self.peer_id} - Aggregazione modello ricevuto...")
        aggregated_params = aggregate_models([serialize_model_params(self.model), received_params])
        deserialize_model_params(self.model, aggregated_params)
        print(f"Peer {self.peer_id} - Modello aggiornato")

    def run(self):
        for _ in range(self.total_rounds):
            if self.stop_flag:
                break
            self.train_one_epoch()
            self.share_model()
            self.test_model(record=True)
            self.round += 1
            self.round_barrier.wait()

    def start(self):
        print(f"Peer {self.peer_id} - Avvio thread...")
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join()

def run_simulation(num_peers, num_segments, total_rounds, dataset_type):
    # Seleziona la funzione di caricamento dati e la rete adeguata in base al dataset scelto
    if dataset_type == "mnist":
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "cifar10":
        load_fn = load_data_cifar10
        NetClass = NetCIFAR
        net_params = {"in_channels": 3, "input_size": (32, 32), "num_classes": 10}
    elif dataset_type == "cifar100":
        load_fn = load_data_cifar100
        NetClass = NetCIFAR
        net_params = {"in_channels": 3, "input_size": (32, 32), "num_classes": 100}
    elif dataset_type == "fmnist":
        load_fn = load_data_fmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "stl10":
        load_fn = load_data_stl10
        NetClass = NetSTL
        net_params = {"in_channels": 3, "input_size": (96, 96), "num_classes": 10}
    elif dataset_type == "imagenet":
        load_fn = load_data_imagenet
        NetClass = NetImageNet
        net_params = {"in_channels": 3, "input_size": (224, 224), "num_classes": 1000}
    else:
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}

    datasets, test_dataset = load_fn(num_peers)
    topology = create_topology(num_peers, num_segments)
    print("Topologia di rete:")
    for peer_id, neighbors in topology.items():
        print(f"Peer {peer_id}: {neighbors}")
    round_barrier = threading.Barrier(num_peers)
    peers_dict = {i: None for i in range(num_peers)}
    for i in range(num_peers):
        account = accounts[i]
        peer = Peer(i, datasets[i], test_dataset, topology, peers_dict, account, total_rounds, round_barrier, NetClass, net_params)
        peers_dict[i] = peer
        receipt = register_client("admin", "admin", account)
        if receipt == "already_registered":
            print(f"Peer {i} has already registered with account {account}")
        elif receipt:
            print(f"Peer {i} registrato con successo con l'account {account}")
        else:
            print(f"Peer {i} non è stato registrato (o c'è stato un errore) con l'account {account}")
    reset_receipt = reset_tokens(accounts[0])
    if reset_receipt:
        print("Reset dei token completato con successo.")
    else:
        print("Errore nel reset dei token.")
    for peer in peers_dict.values():
        peer.start()
    for peer in peers_dict.values():
        peer.thread.join()
    peers_accuracies = {peer_id: peer.accuracy_history for peer_id, peer in peers_dict.items()}
    peers_losses = {peer_id: peer.loss_history for peer_id, peer in peers_dict.items()}
    plots.plot_accuracy(peers_accuracies)
    plots.plot_loss(peers_losses)
    plots.plot_final_accuracy_bar(peers_accuracies)
    plots.plot_combined(peers_accuracies, peers_losses)

if __name__ == "__main__":
    run_simulation(num_peers=10, num_segments=2, total_rounds=10, dataset_type="cifar10")










