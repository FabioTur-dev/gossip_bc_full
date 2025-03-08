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
from blockchain_functions import register_client, save_hash, check_hash_exists, penalize_client, reset_tokens
import plots2
from nets import NetMNIST, NetCIFAR
from sklearn.cluster import KMeans

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


class DirichletPartitioner:
    def __init__(self, dataset, n_peers, beta):
        """
        Partition dataset using Dirichlet distribution.
        Args:
            dataset: PyTorch dataset con attributo targets.
            n_peers: Numero di client/peer.
            beta: Parametro di concentrazione per la distribuzione di Dirichlet.
        """
        self.dataset = dataset
        self.n_peers = n_peers
        self.beta = beta
        # Per dataset come CIFAR10, dataset.targets è una lista, quindi la convertiamo in tensore.
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
    for i, ds in enumerate(client_datasets):
        print(f"Client {i}: {len(ds)} samples")
    return client_datasets, test_dataset


def load_data_cifar10(n_peers, beta=0.5):
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
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


def create_clustered_topology(client_datasets, num_segments):
    """
    Raggruppa i peer in base alla distribuzione delle label e crea la topologia
    per segmented gossip learning.
    I peer all'interno dello stesso cluster comunicheranno tra loro.
    """
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
    for peer_id, ds in enumerate(client_datasets):
        counts = np.zeros(n_classes, dtype=np.float32)
        for idx in ds.indices:
            lbl = int(dataset_ref.targets[idx])
            counts[lbl] += 1
        total = counts.sum()
        dist = counts / total if total > 0 else np.zeros_like(counts)
        label_dists.append(dist)
    label_dists = np.array(label_dists)

    # Eseguiamo il clustering con K-Means
    kmeans = KMeans(n_clusters=num_segments, random_state=42)
    cluster_labels = kmeans.fit_predict(label_dists)

    segments_list = [[] for _ in range(num_segments)]
    for peer_id in range(num_peers):
        c = cluster_labels[peer_id]
        segments_list[c].append(peer_id)

    topology = {}
    for seg in segments_list:
        for p in seg:
            topology[p] = [x for x in seg if x != p]

    print("Segments (clusterizzati):")
    for idx, seg in enumerate(segments_list):
        print(f"  Segment {idx}: {seg}")

    # Per consentire anche una comunicazione occasionale inter-segmento,
    # aggiungiamo in ogni peer un "inter-segment neighbor" scelto casualmente
    # dai peer non appartenenti al suo cluster.
    all_peers = set(range(num_peers))
    for p in range(num_peers):
        current_cluster = set(topology[p] + [p])
        others = list(all_peers - current_cluster)
        if others:
            # Con una probabilità (es. 10%) aggiungiamo un inter-segment neighbor
            if random.random() < 0.1:
                inter_neighbor = random.choice(others)
                topology[p].append(inter_neighbor)
    return topology


class AsyncPeer:
    def __init__(self, peer_id, dataset, test_dataset, topology, peers_dict, account,
                 simulation_time, NetClass, net_params, event_log):
        self.peer_id = peer_id
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.model = NetClass(**net_params).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
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
        self.dp_clip_threshold = 100.0
        self.dp_noise_multiplier = 0

    def train_one_epoch(self):
        self.model.train()
        trainloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        total_loss = 0
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Starting training (count {self.train_count + 1}) at time {timestamp:.2f}s")
        for data, target in trainloader:
            if self.stop_flag:
                break
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

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
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * self.dp_noise_multiplier * self.dp_clip_threshold
                    p.grad.data.add_(noise)
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(trainloader)
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
        print(f"Peer {self.peer_id} - Training loss: {avg_loss:.4f} at time {end_timestamp:.2f}s")
        return avg_loss

    def test_model(self):
        self.model.eval()
        testloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Starting testing at time {timestamp:.2f}s")
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
        print(f"Peer {self.peer_id} - Test Accuracy: {accuracy:.2f}% at time {end_timestamp:.2f}s")
        return accuracy

    def save_cid_to_blockchain(self, model_cid):
        receipt = save_hash(model_cid, self.account)
        if receipt:
            print(f"📜 [BLOCKCHAIN] Peer {self.peer_id} - CID saved: {model_cid}")
        else:
            print(f"Peer {self.peer_id} - Error saving CID to blockchain")

    def check_cid(self, model_cid):
        exists = check_hash_exists(model_cid, self.account)
        if exists:
            print(f"✅ [BLOCKCHAIN] Peer {self.peer_id} - CID {model_cid} exists")
        else:
            print(f"😒 [BLOCKCHAIN] Peer {self.peer_id} - CID {model_cid} does NOT exist")
        return exists

    def share_model(self):
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Sharing model at time {timestamp:.2f}s")
        params = serialize_model_params(self.model)
        model_cid = save_model_to_ipfs(params)
        self.event_log.append({
            'peer_id': self.peer_id,
            'event': 'share',
            'timestamp': timestamp,
            'cid': model_cid
        })
        print(f"Peer {self.peer_id} - Model saved to IPFS: {model_cid}")
        self.save_cid_to_blockchain(model_cid)
        self.last_share_time = timestamp
        # Condividi con i vicini intra-cluster
        for neighbor_id in self.neighbors:
            if neighbor_id in self.peers:
                self.peers[neighbor_id].receive_model(self.peer_id, params, model_cid, timestamp)
            else:
                print(f"Peer {self.peer_id} - Error: Peer {neighbor_id} not found")
        # Comunicazione inter-segmento: con probabilità 10% scegliamo un peer a caso fuori dal cluster
        if random.random() < 0.8:
            all_peers = set(self.peers.keys())
            current_cluster = set(self.neighbors + [self.peer_id])
            external_peers = list(all_peers - current_cluster)
            if external_peers:
                inter_peer = random.choice(external_peers)
                print(f"Peer {self.peer_id} - Also sharing model inter-segment with Peer {inter_peer}")
                self.peers[inter_peer].receive_model(self.peer_id, params, model_cid, timestamp)

    def receive_model(self, sender_id, params, model_cid, share_timestamp):
        receive_timestamp = time.time()
        print(f"Peer {self.peer_id} - Received model from Peer {sender_id} with CID: {model_cid} at time {receive_timestamp:.2f}s")
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
            print(f"Peer {self.peer_id} - Rejecting model from Peer {sender_id}: CID not verified")
            return
        self.aggregate_received_model(params)

    def aggregate_received_model(self, received_params):
        timestamp = time.time()
        print(f"Peer {self.peer_id} - Aggregating received model at time {timestamp:.2f}s")
        current_params = serialize_model_params(self.model)
        aggregated_params = []
        for i in range(len(current_params)):
            weighted_param = 0.5 * current_params[i] + 0.5 * received_params[i]
            aggregated_params.append(weighted_param)
        deserialize_model_params(self.model, aggregated_params)
        print(f"Peer {self.peer_id} - Model updated at time {time.time():.2f}s")

    def run(self):
        start_time = time.time()
        self.test_model()
        while time.time() - start_time < self.simulation_time and not self.stop_flag:
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


def run_simulation(num_peers, num_segments, simulation_time, dataset_type, beta=0.5):
    print(f"Starting asynchronous gossip learning simulation with {num_peers} peers")
    print(f"Using Dirichlet partitioning with beta={beta} ({beta} -> higher means more IID data)")
    if dataset_type == "mnist":
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "cifar10":
        load_fn = load_data_cifar10
        NetClass = NetCIFAR
        net_params = {"in_channels": 3, "input_size": (32, 32), "num_classes": 10}
    elif dataset_type == "emnist":
        load_fn = load_data_emnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "fmnist":
        load_fn = load_data_fmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "kmnist":
        load_fn = load_data_kmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    elif dataset_type == "qmnist":
        load_fn = load_data_qmnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}
    else:
        print(f"Unknown dataset type: {dataset_type}, defaulting to MNIST")
        load_fn = load_data_mnist
        NetClass = NetMNIST
        net_params = {"in_channels": 1, "input_size": (28, 28), "num_classes": 10}

    # Carica i dataset partizionati tramite Dirichlet
    datasets, test_dataset = load_fn(num_peers, beta)

    # Crea la topologia basata sul clustering delle distribuzioni di label
    topology = create_clustered_topology(datasets, num_segments)

    print("Network topology:")
    for peer_id, neighbors in topology.items():
        print(f"Peer {peer_id}: {neighbors}")

    event_log = []
    peers_dict = {}
    for i in range(num_peers):
        account = accounts[i]
        receipt = register_client("admin", "admin", account)
        if receipt == "already_registered":
            print(f"Peer {i} has already registered with account {account}")
        elif receipt:
            print(f"Peer {i} successfully registered with account {account}")
        else:
            print(f"Peer {i} registration failed with account {account}")
        peer = AsyncPeer(
            i, datasets[i], test_dataset, topology, peers_dict,
            account, simulation_time, NetClass, net_params, event_log
        )
        peers_dict[i] = peer

    reset_receipt = reset_tokens(accounts[0])
    if reset_receipt:
        print("Token reset completed successfully")
    else:
        print("Error resetting tokens")

    for peer in peers_dict.values():
        peer.start()

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

    peers_accuracies = {
        peer_id: [(t - simulation_start, acc) for t, acc in peer.accuracy_history]
        for peer_id, peer in peers_dict.items()
    }
    peers_losses = {
        peer_id: [(t - simulation_start, loss) for t, loss in peer.loss_history]
        for peer_id, peer in peers_dict.items()
    }

    plots2.plot_accuracy_over_time(peers_accuracies)
    plots2.plot_loss_over_time(peers_losses)
    plots2.plot_final_accuracy_bar(peers_accuracies)
    plots2.plot_combined(peers_accuracies, peers_losses)
    plots2.plot_events_timeline(event_log, simulation_start, simulation_time)
    plots2.plot_communication_graph(event_log, num_peers, simulation_start)

    print("Simulation completed and plots generated")
    return peers_dict, event_log


if __name__ == "__main__":
    run_simulation(
        num_peers=10,
        num_segments=2,
        simulation_time=600,  # seconds
        dataset_type="cifar10",  # supporta anche "mnist", "emnist", "fmnist", "kmnist", "qmnist"
        beta=1.0  # Dirichlet concentration parameter
    )




