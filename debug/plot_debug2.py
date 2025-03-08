import os
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Usato per il boxplot

# Impostazioni grafiche aggiornate per testi più grandi
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 20,  # dimensione di base del font
    "axes.titlesize": 26,  # dimensione del titolo
    "axes.labelsize": 22,  # dimensione delle etichette degli assi
})


def plot_peer_flow_bar_with_progress(results_dir):
    pattern = os.path.join(results_dir, "*.json")
    json_files = glob.glob(pattern)

    dataset_peer_data = {}

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "accuracy_over_time" not in data:
            continue

        # Estrai il nome del dataset dal filename
        base = os.path.basename(file_path)
        core = base[len("plot_values_"):-len(".json")]
        match = re.match(r"([a-zA-Z]+)([0-9.]+)", core)
        if match:
            dataset = match.group(1)
        else:
            dataset = core

        peers_data = data["accuracy_over_time"]
        for peer_id, peer_info in peers_data.items():
            accuracies = peer_info.get("accuracies", [])
            if not accuracies:
                continue
            initial = accuracies[0]
            final = accuracies[-1]
            flow = final - initial
            # Calcola la progressione: differenza di ogni punto rispetto all'accuracy iniziale
            progress = [val - initial for val in accuracies]
            dataset_peer_data.setdefault(dataset, []).append({
                "peer": peer_id,
                "initial": initial,
                "final": final,
                "flow": flow,
                "progress": progress
            })

    if not dataset_peer_data:
        print("No valid peer flow data found.")
        return

    # Ordina i dataset e prepara una mappatura di colori usando la palette "deep"
    datasets = sorted(dataset_peer_data.keys())
    n_datasets = len(datasets)
    palette = sns.color_palette("deep", n_datasets)
    dataset_to_color = {d: palette[i] for i, d in enumerate(datasets)}

    # Calcola le posizioni dei gruppi sull'asse x
    group_positions = np.arange(n_datasets)
    total_group_width = 0.8

    fig, ax = plt.subplots(figsize=(12, 8))
    # Per ciascun dataset, raggruppa i peer e traccia le barre
    for i, dataset in enumerate(datasets):
        peers = dataset_peer_data[dataset]
        # Ordina i peer in base al loro id (se numerico)
        try:
            peers.sort(key=lambda x: int(x["peer"]))
        except Exception:
            peers.sort(key=lambda x: x["peer"])
        n_peers = len(peers)
        # Calcola gli offset per centrare le barre nel gruppo
        offsets = np.linspace(-total_group_width / 2, total_group_width / 2, n_peers)
        for j, peer in enumerate(peers):
            x = group_positions[i] + offsets[j]
            bar_width = total_group_width / n_peers * 0.9
            # Traccia la barra per il flow
            ax.bar(x, peer["flow"], width=bar_width,
                   color=dataset_to_color[dataset],
                   edgecolor="black", linewidth=1.5)
            # Traccia i marker per ogni punto intermedio (progress)
            x_coords = np.full(len(peer["progress"]), x)
            ax.plot(x_coords, peer["progress"], marker="o", linestyle="-",
                    color="black", markersize=6)

        # Aggiungi la scritta sopra il gruppo, riassumendo i peer
        peer_ids = [p["peer"] for p in peers]
        try:
            sorted_ids = sorted(peer_ids, key=lambda x: int(x))
        except Exception:
            sorted_ids = sorted(peer_ids)
        if sorted_ids:
            label_text = f"Peer {sorted_ids[0]} ... Peer {sorted_ids[-1]}"
            # Posiziona il testo sopra il gruppo con un riquadro
            ax.text(group_positions[i], ax.get_ylim()[1] * 0.98, label_text,
                    ha="center", va="top", fontsize=20, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2"))

    # Aggiungi una linea orizzontale a y=0 per evidenziare i valori negativi
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    # Aggiungi una nota esplicativa in basso a sinistra
    ax.text(0.02, 0.02, "Negative flow indicates a decrease in accuracy",
            transform=ax.transAxes, fontsize=20, color="red", ha="left", va="bottom")

    ax.set_xticks(group_positions)
    ax.set_xticklabels(datasets, fontsize=22)
    ax.set_title("Peer Flow with Accuracy Progress by Dataset", fontsize=26)
    ax.set_ylabel("Flow (Final - Initial Accuracy)", fontsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    plot_peer_flow_bar_with_progress("")

if __name__ == "__main__":
    main()