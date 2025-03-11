import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, EMNIST, KMNIST, QMNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import numpy as np
import random
import threading
import time
import ssl
import math
from scipy.stats import trim_mean
from phe import paillier
from web3 import Web3

from ipfs_utils import save_model_to_ipfs, load_model_from_ipfs
from blockchain_functions_cluster import (
    register_client, save_hash, reset_tokens, deploy_fedclustering, update_cluster_center
)
import plots2
from nets import NetMNIST, NetCIFAR
from sklearn.cluster import KMeans

# Bypass SSL per sviluppo (rimuovere in produzione)
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
ssl._create_default_https_context = lambda: context  # type: ignore

# Seed per riproducibilità
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazione Web3 e indirizzi smart contract
WEB3_PROVIDER = "http://127.0.0.1:8545"
EXT_CONTRACT_ADDRESS = "0x5843b4cbD4Bbb80F4cF6ee2d6812D42ff833B378"  # ExtendedHashStorage
FED_CLUSTERING_ADDRESS = "0xbeB84916920ED1E09d8F4a1CFdd3118cBD4dB164"  # FedClustering

with open("./build/contracts/ExtendedHashStorage.json") as f:
    ext_contract_data = json.load(f)
    EXT_ABI = ext_contract_data["abi"]

web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
ext_contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
accounts = web3.eth.accounts

with open("./build/contracts/FedClustering.json") as f:
    fed_clustering_data = json.load(f)
    FED_CLUSTERING_ABI = fed_clustering_data["abi"]

fed_clustering_contract = None

# --- Paillier per aggregazione sicura (se necessario) ---
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

def secure_aggregate_distributions_HE(distributions, public_key, private_key):
    d = distributions.shape[1]
    encrypted_sum = [public_key.encrypt(float(x)) for x in distributions[0]]
    for i in range(1, distributions.shape[0]):
        for j in range(d):
            encrypted_sum[j] = encrypted_sum[j] + float(distributions[i, j])
    decrypted = np.array([private_key.decrypt(c) for c in encrypted_sum])
    return decrypted / distributions.shape[0]

# --- Partizionamento dei dati con Dirichlet ---
class DirichletPartitioner:
    def __init__(self, dataset, n_peers, beta):
        self.dataset = dataset
        self.n_peers = n_peers
        self.beta = beta
        if hasattr(dataset, 'targets'):
            self.targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
        else:
            self.targets = torch.tensor([target for _, target in dataset])
        self.n_classes = len(torch.unique(self.targets))
        self.client_idxs = self.partition_data()

    def partition_data(self):
        client_idxs = [[] for _ in range(self.n_peers)]
        idxs = np.arange(len(self.dataset))
        idxs_labels = np.vstack((idxs, self.targets.numpy()))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        label_distribution = np.random.dirichlet([self.beta] * self.n_peers, self.n_classes)
        class_idxs = []
        for k in range(self.n_classes):
            idx_k = idxs[idxs_labels[1, :] == k]
            class_idxs.append(idx_k)
        for c, fracs in enumerate(label_distribution):
            for i, idx_fraction in enumerate(fracs):
                if len(class_idxs[c]) == 0:
                    continue
                n_samples = int(idx_fraction * len(class_idxs[c]))
                if n_samples == 0:
                    continue
                samples = np.random.choice(class_idxs[c], n_samples, replace=False)
                client_idxs[i].extend(samples)
                class_idxs[c] = np.setdiff1d(class_idxs[c], samples)
        for c in range(self.n_classes):
            remaining = class_idxs[c]
            if len(remaining) > 0:
                client_idx = np.random.choice(self.n_peers, size=len(remaining))
                for i, idx in enumerate(remaining):
                    client_idxs[client_idx[i]].append(idx)
        return client_idxs

    def get_client_dataset(self, client_id):
        class SubsetDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
            def __len__(self):
                return len(self.indices)
        return SubsetDataset(self.dataset, self.client_idxs[client_id])

def load_data_mnist(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = MNIST("./data", train=False, download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"MNIST data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

def load_data_cifar10(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset = CIFAR10("./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10("./data", train=False, download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"CIFAR-10 data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

def load_data_fmnist(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = FashionMNIST("./data", train=True, download=True, transform=transform)
    test_dataset = FashionMNIST("./data", train=False, download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"Fashion-MNIST data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

def load_data_emnist(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = EMNIST("./data", split="digits", train=True, download=True, transform=transform)
    test_dataset = EMNIST("./data", split="digits", train=False, download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"EMNIST (digits) data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

def load_data_kmnist(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = KMNIST("./data", train=True, download=True, transform=transform)
    test_dataset = KMNIST("./data", train=False, download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"KMNIST data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

def load_data_qmnist(n_peers, beta=0.5):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    dataset = QMNIST("./data", what="train", download=True, transform=transform)
    test_dataset = QMNIST("./data", what="test", download=True, transform=transform)
    partitioner = DirichletPartitioner(dataset, n_peers, beta)
    client_datasets = [partitioner.get_client_dataset(i) for i in range(n_peers)]
    print(f"QMNIST data partitioned with Dirichlet beta={beta}")
    return client_datasets, test_dataset

# --- Serializzazione dei parametri del modello ---
def serialize_model_params(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]

def deserialize_model_params(model, params):
    for i, p in enumerate(model.parameters()):
        p.data = torch.from_numpy(params[i]).to(device)

def robust_aggregate(updates, trim_ratio):
    aggregated = []
    num_params = len(updates[0])
    for i in range(num_params):
        param_updates = [update[i] for update in updates]
        param_stack = np.stack(param_updates, axis=0)
        robust_avg = trim_mean(param_stack, proportiontocut=trim_ratio, axis=0)
        aggregated.append(robust_avg)
    return aggregated

# --- Aggregazione globale eseguita dal leader ---
def global_aggregation(leader_peer, peers_dict, fed_clustering_contract, num_segments):
    """
    Per ogni segmento, aggrega i parametri (solo del layer segmentato) dai peer che hanno quell'assegnazione.
    Il modello globale viene ricostruito a partire dai parametri del leader per le parti non segmentate,
    aggiornando il layer segmentato con la media dei segmenti.
    """
    # Usa i parametri correnti del leader come base
    global_params = serialize_model_params(leader_peer.model)
    # Determina quale layer è segmentato (per MNIST si usa fc3, per CIFAR il layer finale di classifier)
    if hasattr(leader_peer.model, "fc3"):
        segmented_layer = "fc3"
        layer = leader_peer.model.fc3
    elif hasattr(leader_peer.model, "classifier"):
        segmented_layer = "classifier"
        layer = leader_peer.model.classifier[-1]
    else:
        print("Modello non conforme: nessun layer segmentato trovato")
        return global_params

    # Ottieni i parametri originali (assumiamo che siano gli ultimi due: peso e bias)
    # Per ogni segmento, aggrega i parametri dai peer aventi quell'assegnazione
    weight_shape = layer.weight.data.shape
    bias_shape = layer.bias.data.shape if layer.bias is not None else None
    aggregated_weight = np.zeros(weight_shape)
    aggregated_bias = np.zeros(bias_shape) if bias_shape is not None else None

    for segment in range(num_segments):
        # Ottieni i confini del segmento (start, end)
        (start_idx, end_idx) = fed_clustering_contract.functions.getSegmentBoundaries(segment).call()
        segment_peer_params = []
        for peer in peers_dict.values():
            if peer.model_segment == segment:
                params = serialize_model_params(peer.model)
                # Supponiamo che i parametri segmentati siano gli ultimi due della lista
                if segmented_layer in ["fc3", "classifier"]:
                    weight = params[-2]
                    bias = params[-1]
                    segment_peer_params.append((weight[start_idx:end_idx, :], bias[start_idx:end_idx]))
        if segment_peer_params:
            weights_list = [p[0] for p in segment_peer_params]
            biases_list = [p[1] for p in segment_peer_params]
            agg_weight = np.mean(weights_list, axis=0)
            agg_bias = np.mean(biases_list, axis=0)
            aggregated_weight[start_idx:end_idx, :] = agg_weight
            aggregated_bias[start_idx:end_idx] = agg_bias
        else:
            # Se nessun peer è presente per quel segmento, usa i parametri del leader
            aggregated_weight[start_idx:end_idx, :] = layer.weight.data.cpu().numpy()[start_idx:end_idx, :]
            aggregated_bias[start_idx:end_idx] = layer.bias.data.cpu().numpy()[start_idx:end_idx]
    # Sostituisci i parametri segmentati nel modello globale
    global_params[-2] = aggregated_weight
    global_params[-1] = aggregated_bias
    return global_params

# Variabile globale per tenere traccia del CID del modello globale
last_global_cid = None

# --- Classe AsyncPeer aggiornata ---
class AsyncPeer:
    def __init__(self, peer_id, dataset, test_dataset, topology, peers_dict, account,
                 simulation_time, NetClass, net_params, event_log):
        self.peer_id = peer_id
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.model = NetClass(**net_params).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.neighbors = topology[peer_id]
        self.peers = peers_dict
        self.stop_flag = False
        self.account = account
        self.thread = threading.Thread(target=self.run)
        self.event_log = event_log
        self.train_count = 0
        self.accuracy_history = []
        self.loss_history = []
        self.last_share_time = 0
        self.last_test_time = 0
        self.simulation_time = simulation_time
        self.train_interval = max(1, np.random.normal(5, 1))
        self.share_interval = max(1, np.random.normal(8, 2))
        self.test_interval = max(2, np.random.normal(10, 2))
        # Parametri per DP
        self.initial_dp_noise_multiplier = 0.01
        self.final_dp_noise_multiplier = 0.001
        self.dp_noise_multiplier = self.initial_dp_noise_multiplier
        self.received_buffer = []
        self.start_time = None
        # Query al contratto per conoscere il segmento assegnato
        self.model_segment = fed_clustering_contract.functions.getPeerSegment(self.account).call()
        print(f"Peer {self.peer_id} assigned to model segment {self.model_segment} (from blockchain)")
        # Variabili per aggiornamento dal modello globale
        self.last_global_cid_seen = None
        self.global_update_interval = max(5, np.random.normal(15, 5))
        self.last_global_update_time = 0

    def update_dp_noise_multiplier(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        progress = min(elapsed / self.simulation_time, 1.0)
        self.dp_noise_multiplier = self.initial_dp_noise_multiplier - \
            progress * (self.initial_dp_noise_multiplier - self.final_dp_noise_multiplier)
        print(f"Peer {self.peer_id} - DP noise multiplier = {self.dp_noise_multiplier:.6f} (progress: {progress*100:.1f}%)")

    def train_one_epoch(self):
        self.update_dp_noise_multiplier()
        self.model.train()
        trainloader = DataLoader(self.dataset, batch_size=128, shuffle=True)
        total_loss = 0
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Training start (count {self.train_count + 1}) at t={timestamp:.2f}")
        for data, target in trainloader:
            if self.stop_flag:
                break
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping e DP noise
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = 5.0 / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * self.dp_noise_multiplier * 5.0
                    p.grad.data.add_(noise)

            # ---- Logica di segmentazione ----
            if hasattr(self.model, "fc3"):
                layer = self.model.fc3
                (start_idx, end_idx) = fed_clustering_contract.functions.getSegmentBoundaries(self.model_segment).call()
                out_features = layer.out_features
                if layer.weight.grad is not None:
                    for i in range(out_features):
                        if i < start_idx or i >= end_idx:
                            layer.weight.grad.data[i, :].zero_()
                if layer.bias is not None and layer.bias.grad is not None:
                    for i in range(out_features):
                        if i < start_idx or i >= end_idx:
                            layer.bias.grad.data[i].zero_()
            elif hasattr(self.model, "classifier"):
                last_layer = self.model.classifier[-1]
                out_features = last_layer.out_features
                (start_idx, end_idx) = fed_clustering_contract.functions.getSegmentBoundaries(self.model_segment).call()
                if last_layer.weight.grad is not None:
                    for i in range(out_features):
                        if i < start_idx or i >= end_idx:
                            last_layer.weight.grad.data[i, :].zero_()
                if last_layer.bias is not None and last_layer.bias.grad is not None:
                    for i in range(out_features):
                        if i < start_idx or i >= end_idx:
                            last_layer.bias.grad.data[i].zero_()
            # -----------------------------
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(trainloader) if len(trainloader) > 0 else 0
        end_timestamp = time.time()
        self.event_log.append({
            'peer_id': self.peer_id,
            'event': 'train',
            'timestamp': timestamp,
            'end_timestamp': end_timestamp,
            'duration': end_timestamp - timestamp,
            'train_count': self.train_count,
            'loss': avg_loss
        })
        self.loss_history.append((timestamp, avg_loss))
        self.train_count += 1
        print(f"Peer {self.peer_id} - Training loss: {avg_loss:.4f} at t={end_timestamp:.2f}")
        return avg_loss

    def test_model(self):
        self.model.eval()
        testloader = DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Testing start at t={timestamp:.2f}")
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        end_timestamp = time.time()
        self.event_log.append({
            'peer_id': self.peer_id,
            'event': 'test',
            'timestamp': timestamp,
            'end_timestamp': end_timestamp,
            'duration': end_timestamp - timestamp,
            'accuracy': accuracy
        })
        self.accuracy_history.append((timestamp, accuracy))
        self.last_test_time = timestamp
        print(f"Peer {self.peer_id} - Test Accuracy: {accuracy:.2f}% at t={end_timestamp:.2f}")
        return accuracy

    def save_cid_to_blockchain(self, model_cid):
        receipt = save_hash(model_cid, self.account)
        if receipt:
            print(f"Peer {self.peer_id} - CID saved: {model_cid}")
        else:
            print(f"Peer {self.peer_id} - Error saving CID to blockchain")

    # La funzione receive_model ora non valida più ogni singolo update; la validazione avverrà solo sul modello globale
    def receive_model(self, sender_id, params, model_cid, share_timestamp):
        receive_timestamp = time.time()
        print(f"Peer {self.peer_id} - Received model from Peer {sender_id} at t={receive_timestamp:.2f}")
        self.event_log.append({
            'peer_id': self.peer_id,
            'event': 'receive',
            'timestamp': receive_timestamp,
            'share_timestamp': share_timestamp,
            'sender_id': sender_id,
            'cid': model_cid,
            'latency': receive_timestamp - share_timestamp
        })
        self.aggregate_received_model(params)

    def aggregate_received_model(self, received_params):
        self.received_buffer.append(received_params)
        if len(self.received_buffer) >= 2:
            current_params = serialize_model_params(self.model)
            updates = [current_params] + self.received_buffer
            new_params = robust_aggregate(updates, trim_ratio=0.1)
            deserialize_model_params(self.model, new_params)
            print(f"Peer {self.peer_id} - Aggregated with {len(updates)} updates at t={time.time():.2f}")
            self.received_buffer = []

    def share_model(self):
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Sharing model at t={timestamp:.2f}")
        params = serialize_model_params(self.model)
        model_cid = save_model_to_ipfs(params)
        self.event_log.append({'peer_id': self.peer_id, 'event': 'share', 'timestamp': timestamp, 'cid': model_cid})
        print(f"Peer {self.peer_id} - Model saved to IPFS: {model_cid}")
        self.save_cid_to_blockchain(model_cid)
        self.last_share_time = timestamp
        # La logica di gossip locale può rimanere, ma non si esegue la validazione hash ad ogni scambio
        for neighbor_id in self.neighbors:
            if neighbor_id in self.peers:
                self.peers[neighbor_id].receive_model(self.peer_id, params, model_cid, timestamp)
            else:
                print(f"Peer {self.peer_id} - Error: Peer {neighbor_id} not found")

    def update_from_global_model(self):
        global last_global_cid
        if last_global_cid is not None and last_global_cid != self.last_global_cid_seen:
            print(f"Peer {self.peer_id} updating from global model CID {last_global_cid}")
            global_params = load_model_from_ipfs(last_global_cid)
            if global_params is None:
                print(f"Peer {self.peer_id} - Error loading global model from IPFS")
                return
            (start_idx, end_idx) = fed_clustering_contract.functions.getSegmentBoundaries(self.model_segment).call()
            # Aggiorna la porzione segmentata del layer finale
            if hasattr(self.model, "fc3"):
                local_weight = self.model.fc3.weight.data.cpu().numpy()
                local_bias = self.model.fc3.bias.data.cpu().numpy()
                local_weight[start_idx:end_idx, :] = global_params[-2][start_idx:end_idx, :]
                local_bias[start_idx:end_idx] = global_params[-1][start_idx:end_idx]
                self.model.fc3.weight.data = torch.from_numpy(local_weight).to(device)
                self.model.fc3.bias.data = torch.from_numpy(local_bias).to(device)
            elif hasattr(self.model, "classifier"):
                local_weight = self.model.classifier[-1].weight.data.cpu().numpy()
                local_bias = self.model.classifier[-1].bias.data.cpu().numpy()
                local_weight[start_idx:end_idx, :] = global_params[-2][start_idx:end_idx, :]
                local_bias[start_idx:end_idx] = global_params[-1][start_idx:end_idx]
                self.model.classifier[-1].weight.data = torch.from_numpy(local_weight).to(device)
                self.model.classifier[-1].bias.data = torch.from_numpy(local_bias).to(device)
            self.last_global_cid_seen = last_global_cid

    def run(self):
        self.start_time = time.time()
        self.test_model()
        while time.time() - self.start_time < self.simulation_time and not self.stop_flag:
            current_time = time.time()
            # Training
            time_since_last_train = current_time - (self.loss_history[-1][0] if self.loss_history else 0)
            if not self.loss_history or time_since_last_train >= self.train_interval:
                self.train_one_epoch()
            # Condivisione locale
            time_since_last_share = current_time - self.last_share_time
            if time_since_last_share >= self.share_interval:
                self.share_model()
            # Testing periodico
            time_since_last_test = current_time - self.last_test_time
            if time_since_last_test >= self.test_interval:
                self.test_model()
            # Aggiornamento dal modello globale
            if current_time - self.last_global_update_time >= self.global_update_interval:
                self.update_from_global_model()
                self.last_global_update_time = current_time
            time.sleep(0.2)

    def start(self):
        print(f"Peer {self.peer_id} - Starting thread...")
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread.is_alive():
            self.thread.join()
            print(f"Peer {self.peer_id} - Thread stopped")

# --- Creazione di una topologia casuale per il gossip ---
def create_random_topology(num_peers, average_degree=3, seed=42):
    random.seed(seed)
    topology = {p: [] for p in range(num_peers)}
    possible_edges = []
    for i in range(num_peers):
        for j in range(i+1, num_peers):
            possible_edges.append((i, j))
    random.shuffle(possible_edges)
    desired_links = int((num_peers * average_degree) / 2)
    edges_added = 0
    for (a, b) in possible_edges:
        if edges_added >= desired_links:
            break
        topology[a].append(b)
        topology[b].append(a)
        edges_added += 1
    return topology

# --- Pubblicazione dei centri di clustering (come da vecchia logica) ---
def publish_cluster_centers(client_datasets, num_segments, fedclustering_address, owner_account, scale=1000):
    num_peers = len(client_datasets)
    dataset_ref = client_datasets[0].dataset
    global_max_label = 0
    for ds in client_datasets:
        for idx in ds.indices:
            lbl = int(dataset_ref.targets[idx])
            if lbl > global_max_label:
                global_max_label = lbl
    n_classes = global_max_label + 1

    label_dists = []
    for ds in client_datasets:
        counts = np.zeros(n_classes, dtype=np.float32)
        for idx in ds.indices:
            lbl = int(dataset_ref.targets[idx])
            counts[lbl] += 1
        total = counts.sum()
        dist = counts / total if total > 0 else np.zeros_like(counts)
        label_dists.append(dist)
    label_dists = np.array(label_dists)

    kmeans = KMeans(n_clusters=num_segments, init="k-means++", random_state=42)
    kmeans.fit(label_dists)
    cluster_centers = kmeans.cluster_centers_

    for clusterId, center_vec in enumerate(cluster_centers):
        center_int = [int(round(x * scale)) for x in center_vec]
        receipt = update_cluster_center(fedclustering_address, clusterId, center_int, owner_account)
        print(f"Cluster {clusterId} center published on blockchain: {center_int}, receipt: {receipt}")

    print("All cluster centers published.")
    published_centers = []
    for center in cluster_centers:
        published_centers.append([int(round(x * scale)) for x in center])
    return published_centers

# --- Main simulation ---
def run_simulation(num_peers, num_segments, simulation_time, dataset_type, beta=0.5):
    print(f"Starting asynchronous gossip learning simulation with {num_peers} peers")
    print(f"Using Dirichlet partitioning with beta={beta}")

    if dataset_type == "mnist":
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}
    elif dataset_type == "cifar10":
        load_fn = load_data_cifar10
        NetClass = NetCIFAR
        net_params = {"in_channels": 3, "input_size": (32,32), "num_classes": 10}
    elif dataset_type == "emnist":
        load_fn = load_data_emnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}
    elif dataset_type == "fmnist":
        load_fn = load_data_fmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}
    elif dataset_type == "kmnist":
        load_fn = load_data_kmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}
    elif dataset_type == "qmnist":
        load_fn = load_data_qmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}
    else:
        print(f"Unknown dataset type: {dataset_type}, defaulting to MNIST")
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28,28), "num_classes": 10}

    datasets, test_dataset = load_fn(num_peers, beta)
    dataset_ref = datasets[0].dataset
    n_classes = len(torch.unique(torch.tensor(dataset_ref.targets)))

    # Deploy del contratto FedClustering
    fedclustering_address = deploy_fedclustering(n_classes, num_segments, accounts[0])
    print("FedClustering address:", fedclustering_address)
    global FED_CLUSTERING_ADDRESS
    FED_CLUSTERING_ADDRESS = fedclustering_address

    # Inizializza il contratto FedClustering
    global fed_clustering_contract
    fed_clustering_contract = web3.eth.contract(
        address=Web3.to_checksum_address(FED_CLUSTERING_ADDRESS), abi=FED_CLUSTERING_ABI
    )

    # Imposta on-chain la dimensione del layer di output
    tx = fed_clustering_contract.functions.setOutputLayerSize(10).transact({'from': accounts[0]})
    web3.eth.wait_for_transaction_receipt(tx)
    print("Output layer size set to 10 on-chain.")

    # 1) Pubblica i cluster centers (on-chain)
    cluster_centers = publish_cluster_centers(datasets, num_segments, FED_CLUSTERING_ADDRESS, accounts[0], scale=1000)

    # 2) Assegna a ogni peer un segmento on-chain (criterio circolare)
    for i in range(num_peers):
        segment = i % num_segments
        tx = fed_clustering_contract.functions.assignPeerSegment(accounts[i], segment).transact({'from': accounts[0]})
        web3.eth.wait_for_transaction_receipt(tx)
        print(f"Peer {i} assigned segment {segment} on-chain.")

    # 3) Crea topologia casuale per il gossip
    topology = create_random_topology(num_peers, average_degree=3, seed=42)
    print("Random topology created:")
    for peer_id, neighbors in topology.items():
        print(f"Peer {peer_id}: {neighbors}")

    # Crea e registra i peer
    event_log = {}
    peers_dict = {}
    for i in range(num_peers):
        account = accounts[i]
        receipt = register_client("admin", "admin", account)
        if receipt == "already_registered":
            print(f"Peer {i} already registered with account {account}")
        elif receipt:
            print(f"Peer {i} registered with account {account}")
        else:
            print(f"Peer {i} registration failed, account {account}")

        event_log[i] = []
        peer = AsyncPeer(i, datasets[i], test_dataset, topology, peers_dict,
                         account, simulation_time, NetClass, net_params, event_log[i])
        peers_dict[i] = peer

    reset_receipt = reset_tokens(accounts[0])
    if reset_receipt:
        print("Token reset completed successfully")
    else:
        print("Error resetting tokens")

    for peer in peers_dict.values():
        peer.start()

    # Variabile per gestire la global aggregation
    global_agg_interval = simulation_time / 10  # ad esempio, ogni 10% della simulazione
    simulation_start = time.time()
    last_global_agg_time = simulation_start

    try:
        while time.time() - simulation_start < simulation_time:
            elapsed = time.time() - simulation_start
            print(f"Simulation progress: {elapsed:.1f}/{simulation_time} seconds")
            # Verifica se è il momento di eseguire l'aggregazione globale
            if time.time() - last_global_agg_time >= global_agg_interval:
                # Elezione del leader tramite smart contract
                leader_address = fed_clustering_contract.functions.electLeader().call()
                leader_peer = None
                for peer in peers_dict.values():
                    if peer.account == leader_address:
                        leader_peer = peer
                        break
                if leader_peer is None:
                    print("Leader non trovato fra i peer")
                else:
                    print(f"Global aggregation triggered, leader: {leader_address}")
                    global_params = global_aggregation(leader_peer, peers_dict, fed_clustering_contract, num_segments)
                    global_cid = save_model_to_ipfs(global_params)
                    if global_cid:
                        save_hash(global_cid, leader_peer.account)
                        print("Global model published with CID", global_cid)
                        global last_global_cid
                        last_global_cid = global_cid
                last_global_agg_time = time.time()
            time.sleep(simulation_time / 10)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        for peer in peers_dict.values():
            peer.stop()

    peers_accuracies = {
        pid: [(t - simulation_start, acc) for (t, acc) in peer.accuracy_history]
        for pid, peer in peers_dict.items()
    }
    peers_losses = {
        pid: [(t - simulation_start, loss) for (t, loss) in peer.loss_history]
        for pid, peer in peers_dict.items()
    }

    plots2.plot_accuracy_over_time(peers_accuracies)
    plots2.plot_loss_over_time(peers_losses)
    plots2.plot_final_accuracy_bar(peers_accuracies)
    plots2.plot_combined(peers_accuracies, peers_losses)

    combined_event_log = []
    for pid, evts in event_log.items():
        combined_event_log.extend(evts)
    segment_assignment = {pid: peer.model_segment for pid, peer in peers_dict.items()}

    plots2.plot_events_timeline(combined_event_log, simulation_start, simulation_time)
    plots2.plot_communication_graph(combined_event_log, num_peers, simulation_start, segment_assignment)

    print("Simulation completed and plots generated")
    return peers_dict, combined_event_log

if __name__ == "__main__":
    run_simulation(num_peers=10, num_segments=2, simulation_time=1000, dataset_type="mnist", beta=0.1)












