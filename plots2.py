import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import os, json

sns.set_theme(style="whitegrid")

# Encoder custom per convertire tipi NumPy in tipi nativi Python
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_plot_data(plot_key, data):
    # Crea la cartella "results" se non esiste
    os.makedirs("results", exist_ok=True)
    json_file = os.path.join("results", "mnist0.1.json")
    # Prova a caricare i dati esistenti; se fallisce, inizializza con un dizionario vuoto
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                all_data = json.load(f)
        except json.JSONDecodeError:
            all_data = {}
    else:
        all_data = {}
    all_data[plot_key] = data
    with open(json_file, "w") as f:
        json.dump(all_data, f, indent=2, cls=NumpyEncoder)

def plot_accuracy_over_time(peers_accuracies):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("bright", len(peers_accuracies))
    data_dict = {}
    for i, (peer_id, accuracy_data) in enumerate(peers_accuracies.items()):
        if not accuracy_data:
            continue
        times, accuracies = zip(*accuracy_data)
        data_dict[str(peer_id)] = {"times": list(times), "accuracies": list(accuracies)}
        plt.plot(times, accuracies, marker="o", color=palette[i], label=f'Peer {peer_id}')
    plt.title('Test Accuracy Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "accuracy_over_time.png"))
    plt.close()
    save_plot_data("accuracy_over_time", data_dict)

def plot_loss_over_time(peers_losses):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("deep", len(peers_losses))
    data_dict = {}
    for i, (peer_id, loss_data) in enumerate(peers_losses.items()):
        if not loss_data:
            continue
        times, losses = zip(*loss_data)
        data_dict[str(peer_id)] = {"times": list(times), "losses": list(losses)}
        plt.plot(times, losses, marker="o", color=palette[i], label=f'Peer {peer_id}')
    plt.title('Training Loss Over Time', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right')
    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "loss_over_time.png"))
    plt.close()
    save_plot_data("loss_over_time", data_dict)

def plot_final_accuracy_bar(peers_accuracies):
    os.makedirs("results", exist_ok=True)
    final_accuracies = {}
    for peer_id, accuracy_data in peers_accuracies.items():
        if accuracy_data:
            final_accuracies[str(peer_id)] = accuracy_data[-1][1]
        else:
            final_accuracies[str(peer_id)] = 0.0
    peers = list(final_accuracies.keys())
    accuracies = [final_accuracies[peer] for peer in peers]
    plt.figure(figsize=(10, 6))
    plt.bar(peers, accuracies, color='skyblue')
    plt.title('Final Test Accuracy per Peer', fontsize=16)
    plt.xlabel('Peer ID', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0.1, 100)
    plt.yscale("log", nonpositive='clip')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "final_accuracy_bar.png"))
    plt.close()
    save_plot_data("final_accuracy_bar", final_accuracies)

def plot_combined(peers_accuracies, peers_losses):
    os.makedirs("results", exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    data_dict = {"accuracy": {}, "loss": {}}
    palette1 = sns.color_palette("bright", len(peers_accuracies))
    for i, (peer_id, accuracy_data) in enumerate(peers_accuracies.items()):
        if not accuracy_data:
            continue
        times, accuracies = zip(*accuracy_data)
        data_dict["accuracy"][str(peer_id)] = {"times": list(times), "accuracies": list(accuracies)}
        axs[0].plot(times, accuracies, marker="o", color=palette1[i], label=f'Peer {peer_id}')
    axs[0].set_title('Test Accuracy Over Time', fontsize=16)
    axs[0].set_ylabel('Accuracy (%)', fontsize=14)
    axs[0].set_ylim(0.1, 100)
    axs[0].legend(loc='lower right')
    axs[0].set_xscale("log", nonpositive='clip')
    axs[0].set_yscale("log", nonpositive='clip')

    palette2 = sns.color_palette("deep", len(peers_losses))
    for i, (peer_id, loss_data) in enumerate(peers_losses.items()):
        if not loss_data:
            continue
        times, losses = zip(*loss_data)
        data_dict["loss"][str(peer_id)] = {"times": list(times), "losses": list(losses)}
        axs[1].plot(times, losses, marker="o", color=palette2[i], label=f'Peer {peer_id}')
    axs[1].set_title('Training Loss Over Time', fontsize=16)
    axs[1].set_xlabel('Time (s)', fontsize=14)
    axs[1].set_ylabel('Loss', fontsize=14)
    axs[1].legend(loc='upper right')
    axs[1].set_xscale("log", nonpositive='clip')
    axs[1].set_yscale("log", nonpositive='clip')

    plt.tight_layout()
    plt.savefig(os.path.join("results", "combined_accuracy_loss.png"))
    plt.close()
    save_plot_data("combined_accuracy_loss", data_dict)

def plot_events_timeline(event_log, simulation_start, simulation_time):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(14, 7))
    event_types = ['train', 'test', 'share', 'receive', 'aggregate']
    colors = {'train': 'blue', 'test': 'green', 'share': 'orange', 'receive': 'red', 'aggregate': 'purple'}
    data_points = []
    for event in event_log:
        ts = event['timestamp'] - simulation_start
        evt = event['event']
        peer_id = event['peer_id']
        data_points.append({"time": ts, "peer_id": peer_id, "event": evt})
        plt.scatter(ts, peer_id, color=colors.get(evt, 'black'),
                    label=evt if evt not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.title('Events Timeline', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Peer ID', fontsize=14)
    plt.xlim(0, simulation_time)
    plt.legend()
    plt.xscale("log", nonpositive='clip')
    plt.yscale("log", nonpositive='clip')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "events_timeline.png"))
    plt.close()
    save_plot_data("events_timeline", {"events": data_points, "simulation_start": simulation_start, "simulation_time": simulation_time})

def plot_communication_graph(event_log, num_peers, simulation_start, segment_assignment):
    os.makedirs("results", exist_ok=True)
    G = nx.DiGraph()
    for i in range(num_peers):
        G.add_node(i)

    for event in event_log:
        if event['event'] == 'share':
            sender = event['peer_id']
            cid = event.get('cid', '')
            for evt in event_log:
                if evt['event'] == 'receive' and evt.get('cid', '') == cid:
                    receiver = evt['peer_id']
                    latency = evt.get('latency', 0)
                    if G.has_edge(sender, receiver):
                        G[sender][receiver]['weight'] += latency
                        G[sender][receiver]['count'] += 1
                    else:
                        G.add_edge(sender, receiver, weight=latency, count=1)

    pos = nx.spring_layout(G, seed=42)
    unique_segments = sorted(set(segment_assignment.values()))
    palette = sns.color_palette("bright", len(unique_segments))
    segment_colors = {seg: palette[i] for i, seg in enumerate(unique_segments)}
    node_colors = [segment_colors[segment_assignment[node]] for node in G.nodes()]
    edge_labels = {(u, v): f"{d['weight'] / d['count']:.2f}s" for u, v, d in G.edges(data=True)}

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title('Communication Graph Among Peers\n(Color indicates segment assignment)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "communication_graph.png"))
    plt.close()
    graph_data = {
        "nodes": list(G.nodes()),
        "edges": [
            {"source": u, "target": v, "weight": d["weight"], "count": d["count"]}
            for u, v, d in G.edges(data=True)
        ],
        "segment_assignment": segment_assignment
    }
    save_plot_data("communication_graph", graph_data)



