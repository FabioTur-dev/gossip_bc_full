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
from web3 import Web3
from ipfs_utils import save_model_to_ipfs
from blockchain_functions_cluster import (
    register_client, save_hash, check_hash_exists, penalize_client, reset_tokens,
    deploy_fedclustering, update_cluster_center
)
import plots2
from nets import NetMNIST, NetCIFAR
from sklearn.cluster import KMeans
import ssl
from scipy.stats import trim_mean
from phe import paillier

# Bypass SSL per development (remove in production)
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE
ssl._create_default_https_context = lambda: context  # type: ignore

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define provider and contract addresses
WEB3_PROVIDER = "http://127.0.0.1:8545"
EXT_CONTRACT_ADDRESS = "0xf5c94f8BCEC7FD286901e73C64658ed0670153dA"  # ExtendedHashStorage
FED_CLUSTERING_ADDRESS = "0xc729A5D6dE82B076A0A028Be1405F3A730fAC42b"  # FedClustering

with open("./build/contracts/ExtendedHashStorage.json") as f:
    ext_contract_data = json.load(f)
    EXT_ABI = ext_contract_data["abi"]

web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
ext_contract = web3.eth.contract(address=Web3.to_checksum_address(EXT_CONTRACT_ADDRESS), abi=EXT_ABI)
accounts = web3.eth.accounts

# --- Paillier key generation for secure aggregation in clustering ---
public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

def secure_aggregate_distributions_HE(distributions, public_key, private_key):
    """
    Performs secure aggregation using Paillier homomorphic encryption.
    Each value is cast to float to ensure proper precision.
    """
    d = distributions.shape[1]
    # Encrypt first vector
    encrypted_sum = [public_key.encrypt(float(x)) for x in distributions[0]]
    # Homomorphically add the rest of the vectors
    for i in range(1, distributions.shape[0]):
        for j in range(d):
            encrypted_sum[j] = encrypted_sum[j] + float(distributions[i, j])
    decrypted = np.array([private_key.decrypt(c) for c in encrypted_sum])
    return decrypted / distributions.shape[0]

# --- Data partitioning via Dirichlet ---
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
        # Se rimangono campioni residui li assegniamo in modo casuale
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

# --- Model parameter serialization ---
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

def robust_aggregate(updates, trim_ratio):
    aggregated = []
    num_params = len(updates[0])
    for i in range(num_params):
        param_updates = [update[i] for update in updates]
        param_stack = np.stack(param_updates, axis=0)
        robust_avg = trim_mean(param_stack, proportiontocut=trim_ratio, axis=0)
        aggregated.append(robust_avg)
    return aggregated

# ----------------------------------------------------------------
# New approach for cluster centers publication (federated analytics)
# ----------------------------------------------------------------

def publish_cluster_centers(client_datasets, num_segments, fedclustering_address, owner_account, scale=1000):
    """
    Esegue KMeans sulle distribuzioni di label dei client
    e pubblica solo i centri dei cluster sulla blockchain.
    NON pubblica la mappatura peer->cluster.
    """

    num_peers = len(client_datasets)
    # Scopriamo il numero di classi cercando la max label
    dataset_ref = client_datasets[0].dataset
    global_max_label = 0
    for ds in client_datasets:
        for idx in ds.indices:
            lbl = int(dataset_ref.targets[idx])
            if lbl > global_max_label:
                global_max_label = lbl
    n_classes = global_max_label + 1

    # Calcoliamo le label distribution per ciascun peer
    label_dists = []
    for peer_id, ds in enumerate(client_datasets):
        counts = np.zeros(n_classes, dtype=np.float32)
        for idx in ds.indices:
            lbl = int(dataset_ref.targets[idx])
            counts[lbl] += 1
        total = counts.sum()
        dist = counts / total if total > 0 else np.zeros_like(counts)
        label_dists.append(dist)
    label_dists = np.array(label_dists)

    # KMeans per trovare i centri
    kmeans = KMeans(n_clusters=num_segments, init="k-means++", random_state=42)
    kmeans.fit(label_dists)
    # I centri dei cluster
    cluster_centers = kmeans.cluster_centers_

    # Aggregazione sicura (opzionale): potremmo anche usare Paillier,
    # ma qui ipotizziamo di mediare i centroid su catena. In questo esempio
    # usiamo direttamente i centri trovati da KMeans come "valori pubblicati".
    # Se volessi, potresti fare 'center = secure_aggregate_distributions_HE(...)'
    # ma KMeans interno fa già la media, quindi qui non la stiamo usando.

    # Scaliamo e pubblichiamo i centri
    for clusterId, center_vec in enumerate(cluster_centers):
        center_int = [int(round(x * scale)) for x in center_vec]
        receipt = update_cluster_center(fedclustering_address, clusterId, center_int, owner_account)
        print(f"Cluster {clusterId} center published on blockchain: {center_int}, receipt: {receipt}")

    print("All cluster centers published.")


def create_random_topology(num_peers, average_degree=3, seed=42):
    """
    Crea una topologia random fra i peer.
    - average_degree: numero medio di connessioni per peer
    - seed: per riproducibilità
    """
    random.seed(seed)
    topology = {p: [] for p in range(num_peers)}

    # Collegamenti random, avendo cura di non duplicare e non collegare un peer a sé stesso
    possible_edges = []
    for i in range(num_peers):
        for j in range(i+1, num_peers):
            possible_edges.append((i, j))
    random.shuffle(possible_edges)

    # Numero totale di link desiderati ~ (num_peers * average_degree) / 2
    desired_links = int((num_peers * average_degree) / 2)

    edges_added = 0
    for (a, b) in possible_edges:
        if edges_added >= desired_links:
            break
        # Aggiungiamo collegamento a <-> b
        topology[a].append(b)
        topology[b].append(a)
        edges_added += 1

    return topology

# --- AsyncPeer class for gossip learning ---
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

        # DP
        self.initial_dp_noise_multiplier = 0.01  # iniziale
        self.final_dp_noise_multiplier = 0.001   # finale desiderato
        self.dp_noise_multiplier = self.initial_dp_noise_multiplier
        self.dp_clip_threshold = 5.0

        self.received_buffer = []
        self.start_time = None

        # Pre-calcoliamo la distribuzione locale di label per eventuale "local cluster inference"
        self.local_distribution = self.compute_local_distribution()

    def compute_local_distribution(self):
        dataset_ref = self.dataset.dataset  # la dataset completa
        indices = self.dataset.indices
        # scopriamo quante classi ci sono
        max_label = 0
        for idx in indices:
            lbl = int(dataset_ref.targets[idx])
            if lbl > max_label:
                max_label = lbl
        n_classes = max_label + 1
        counts = np.zeros(n_classes, dtype=np.float32)
        for idx in indices:
            lbl = int(dataset_ref.targets[idx])
            counts[lbl] += 1
        total = counts.sum()
        dist = counts / total if total > 0 else np.zeros_like(counts)
        return dist

    def local_cluster_inference(self, cluster_centers, scale=1000):
        """
        Esempio di inferenza locale del cluster:
        - Otteniamo i centri (list of list) dallo smart contract (qui passati come arg in modo semplificato)
        - Li riscaliamo per tornare allo spazio float
        - Calcoliamo la distanza euclidea dal nostro vettore locale
        - Ritorniamo l'indice del cluster più vicino
        """
        best_cluster = -1
        best_dist = float('inf')
        for cid, center_scaled in enumerate(cluster_centers):
            # Convertiamo da int scaled a float
            center = np.array(center_scaled) / scale
            dist = np.linalg.norm(self.local_distribution - center)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cid
        return best_cluster

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

            # Gradient clipping
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = self.dp_clip_threshold / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)

            # DP noise
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * self.dp_noise_multiplier * self.dp_clip_threshold
                    p.grad.data.add_(noise)

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

    def check_cid(self, model_cid):
        exists = check_hash_exists(model_cid, self.account)
        if exists:
            print(f"Peer {self.peer_id} - CID {model_cid} exists")
        else:
            print(f"Peer {self.peer_id} - CID {model_cid} does NOT exist")
        return exists

    def aggregate_received_model(self, received_params):
        self.received_buffer.append(received_params)
        if len(self.received_buffer) >= 2:
            current_params = serialize_model_params(self.model)
            updates = [current_params] + self.received_buffer
            new_params = robust_aggregate(updates, trim_ratio=0.1)
            deserialize_model_params(self.model, new_params)
            print(f"Peer {self.peer_id} - Aggregated with {len(updates)} updates at t={time.time():.2f}s")
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
        for neighbor_id in self.neighbors:
            if neighbor_id in self.peers:
                self.peers[neighbor_id].receive_model(self.peer_id, params, model_cid, timestamp)
            else:
                print(f"Peer {self.peer_id} - Error: Peer {neighbor_id} not found")

        # Eventuale inter-segment sharing
        if random.random() < 0.8:
            all_peers = set(self.peers.keys())
            current_cluster = set(self.neighbors + [self.peer_id])
            external_peers = list(all_peers - current_cluster)
            if external_peers:
                inter_peer = random.choice(external_peers)
                print(f"Peer {self.peer_id} - Inter-segment sharing with Peer {inter_peer}")
                self.peers[inter_peer].receive_model(self.peer_id, params, model_cid, timestamp)

    def receive_model(self, sender_id, params, model_cid, share_timestamp):
        receive_timestamp = time.time()
        print(f"Peer {self.peer_id} - Received model from Peer {sender_id}, CID={model_cid}, t={receive_timestamp:.2f}")
        self.event_log.append({
            'peer_id': self.peer_id,
            'event': 'receive',
            'timestamp': receive_timestamp,
            'share_timestamp': share_timestamp,
            'sender_id': sender_id,
            'cid': model_cid,
            'latency': receive_timestamp - share_timestamp
        })
        valid = self.check_cid(model_cid)
        if not valid:
            print(f"Peer {self.peer_id} - Model from {sender_id} REJECTED: invalid CID")
            return
        self.aggregate_received_model(params)

    def run(self):
        self.start_time = time.time()
        self.test_model()
        while time.time() - self.start_time < self.simulation_time and not self.stop_flag:
            current_time = time.time()
            time_since_last_train = current_time - (self.loss_history[-1][0] if self.loss_history else 0)
            if not self.loss_history or time_since_last_train >= self.train_interval:
                self.train_one_epoch()
            time_since_last_share = current_time - self.last_share_time
            if time_since_last_share >= self.share_interval:
                self.share_model()
            time_since_last_test = current_time - self.last_test_time
            if time_since_last_test >= self.test_interval:
                self.test_model()
            time.sleep(0.2)

    def start(self):
        print(f"Peer {self.peer_id} - Starting thread...")
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread.is_alive():
            self.thread.join()
            print(f"Peer {self.peer_id} - Thread stopped")

# ----------------------------------------------------------------
# Main simulation
# ----------------------------------------------------------------

def run_simulation(num_peers, num_segments, simulation_time, dataset_type, beta=0.5):
    print(f"Starting asynchronous gossip learning simulation with {num_peers} peers")
    print(f"Using Dirichlet partitioning with beta={beta} ({beta} -> higher means more IID)")

    # Dataset loader
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

    # Caricamento dataset e test set
    datasets, test_dataset = load_fn(num_peers, beta)
    # Scopriamo quante classi reali
    dataset_ref = datasets[0].dataset
    n_classes = len(torch.unique(torch.tensor(dataset_ref.targets)))

    # Deploy del contratto e pubblicazione centri
    fedclustering_address = deploy_fedclustering(n_classes, num_segments, accounts[0])
    print("FedClustering address:", fedclustering_address)

    global FED_CLUSTERING_ADDRESS
    FED_CLUSTERING_ADDRESS = fedclustering_address

    # ------------------------------------------------------
    # 1) Pubblicazione dei centri di cluster su blockchain
    # ------------------------------------------------------
    publish_cluster_centers(datasets, num_segments, FED_CLUSTERING_ADDRESS, accounts[0], scale=1000)

    # ------------------------------------------------------
    # 2) Creazione topologia (random) di gossip
    # ------------------------------------------------------
    topology = create_random_topology(num_peers, average_degree=3, seed=42)
    print("Random topology created:")
    for peer_id, neighbors in topology.items():
        print(f"Peer {peer_id}: {neighbors}")

    # Creazione e registrazione peer
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

        # Inizializzazione peer
        event_log[i] = []
        peer = AsyncPeer(i, datasets[i], test_dataset, topology, peers_dict,
                         account, simulation_time, NetClass, net_params, event_log[i])
        peers_dict[i] = peer

    # Reset token (per sicurezza)
    reset_receipt = reset_tokens(accounts[0])
    if reset_receipt:
        print("Token reset completed successfully")
    else:
        print("Error resetting tokens")

    # Avvio dei thread dei peer
    for peer in peers_dict.values():
        peer.start()

    # Esecuzione della simulazione
    try:
        simulation_start = time.time()
        print(f"Simulation started, will run for {simulation_time} seconds")
        while time.time() - simulation_start < simulation_time:
            elapsed = time.time() - simulation_start
            print(f"Simulation progress: {elapsed:.1f}/{simulation_time} seconds")
            time.sleep(simulation_time / 10)
        print("Simulation time completed")
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        for peer in peers_dict.values():
            peer.stop()

    # Raccolta risultati
    peers_accuracies = {
        pid: [(t - simulation_start, acc) for (t, acc) in peer.accuracy_history]
        for pid, peer in peers_dict.items()
    }
    peers_losses = {
        pid: [(t - simulation_start, loss) for (t, loss) in peer.loss_history]
        for pid, peer in peers_dict.items()
    }

    # Plot finali
    plots2.plot_accuracy_over_time(peers_accuracies)
    plots2.plot_loss_over_time(peers_losses)
    plots2.plot_final_accuracy_bar(peers_accuracies)
    plots2.plot_combined(peers_accuracies, peers_losses)

    # unendo i log di ogni peer in un unico event_log "globale"
    combined_event_log = []
    for pid, evts in event_log.items():
        combined_event_log.extend(evts)
    plots2.plot_events_timeline(combined_event_log, simulation_start, simulation_time)
    plots2.plot_communication_graph(combined_event_log, num_peers, simulation_start)

    print("Simulation completed and plots generated")
    return peers_dict, combined_event_log


if __name__ == "__main__":
    run_simulation(num_peers=10, num_segments=2, simulation_time=1000, dataset_type="mnist", beta=0.1)







