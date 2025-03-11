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


def plot_enhanced_accuracy_lines_for_mnist(results_dir="results"):
    """
    Finds all JSON files in 'results' that contain 'mnist1.0' in the filename.
    For each file, computes the average accuracy over time across all peers (using interpolation)
    and plots a single figure with one line per file.

    The legend displays only the dataset name (without β) and is placed at the bottom right.
    The β value is displayed only once in the center of the plot without a bounding box.
    The area under each line is filled with a very light shading.
    """
    pattern = os.path.join(results_dir, "plot_values_*mnist0.1*.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print("No files found with 'mnist1.0' in the name.")
        return

    plt.figure(figsize=(12, 8))
    beta_value = None  # Per salvare il beta estratto dal primo file processato

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        if "accuracy_over_time" not in data:
            print(f"File {file_path} does not contain 'accuracy_over_time'.")
            continue

        peers_data = data["accuracy_over_time"]

        # Combina i tempi di tutti i peer
        all_times = set()
        for peer in peers_data.values():
            all_times.update(peer.get("times", []))
        if not all_times:
            print(f"File {file_path} has no valid times.")
            continue
        common_times = np.array(sorted(all_times))

        # Interpola le accuracy per ciascun peer sulla timeline comune
        interpolated = []
        for peer in peers_data.values():
            times = np.array(peer["times"])
            accuracies = np.array(peer["accuracies"])
            interp_acc = np.interp(common_times, times, accuracies)
            interpolated.append(interp_acc)
        mean_accuracies = np.mean(interpolated, axis=0)

        # Estrai il dataset e il beta dal nome del file.
        # Atteso: "plot_values_<dataset><beta>.json"
        # Esempio: "plot_values_emnist1.0.json" --> dataset="emnist", beta="1.0"
        base = os.path.basename(file_path)
        core = base[len("plot_values_"):-len(".json")]
        match = re.match(r"([a-zA-Z]+)([0-9.]+)", core)
        if match:
            dataset = match.group(1)
            current_beta = match.group(2)
        else:
            dataset = core
            current_beta = ""
        if beta_value is None and current_beta:
            beta_value = current_beta

        # La legenda mostra solo il nome del dataset (senza β)
        label = dataset

        # Plotta la linea con marker e riempi l'area sottostante con shading leggero
        line, = plt.plot(common_times, mean_accuracies, linewidth=2.5, marker="o", markersize=6, label=label)
        plt.fill_between(common_times, mean_accuracies, alpha=0.05, color=line.get_color())

    plt.title("Average Accuracy Over Time", fontsize=27, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=26)
    plt.ylabel("Accuracy (%)", fontsize=26)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.legend(fontsize=23, loc="lower right")

    # Aggiungi il valore di β al centro del plot (senza riquadro)
    if beta_value:
        plt.text(0.5, 0.5, f"β = {beta_value}", transform=plt.gca().transAxes,
                 fontsize=25, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='none'))

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_accuracy_boxplot_by_dataset(results_dir="results"):
    """
    Creates a boxplot of final accuracies for each dataset.
    For each JSON file (with 'mnist1.0' in the filename), the function extracts the final accuracy
    of each peer (the last value in the "accuracies" list) and groups the data by dataset,
    where the dataset name is extracted from the filename.

    The box colors are set using the same "deep" palette as in the line plot.
    The x-axis shows one label per dataset.
    """
    pattern = os.path.join(results_dir, "*mnist0.1*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        print("No files found with 'mnist1.0' in the name.")
        return

    # Raggruppa le accuracy finali per dataset
    dataset_data = {}
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        if "accuracy_over_time" not in data:
            continue

        # Estrai il nome del dataset
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
            final_acc = accuracies[-1]
            dataset_data.setdefault(dataset, []).append(final_acc)

    if not dataset_data:
        print("No valid accuracy data found.")
        return

    # Ordina i dataset e usa la stessa palette "deep" del line plot
    datasets = sorted(dataset_data.keys())
    color_palette = sns.color_palette("deep", len(datasets))
    dataset_to_color = {d: color_palette[i] for i, d in enumerate(datasets)}

    # Prepara i dati per il boxplot
    plot_data = []
    for d, values in dataset_data.items():
        for val in values:
            plot_data.append({"dataset": d, "final_accuracy": val})
    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x="dataset", y="final_accuracy", data=df, palette=dataset_to_color)
    ax.set_title("Final Accuracy Distribution by Dataset", fontsize=26, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=22)
    ax.set_ylabel("Final Accuracy (%)", fontsize=22)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_peer_flow_bar_with_progress(results_dir="results"):
    pattern = os.path.join(results_dir, "*mnist0.1*.json")
    json_files = glob.glob(pattern)
    if not json_files:
        print("No files found with 'mnist1.0' in the name.")
        return

    # Raccogli i dati per ogni dataset
    # Struttura: { dataset: [ { "peer": peer_id, "initial": float, "final": float, "flow": float, "progress": [delta1, delta2, ...] }, ... ] }
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


def plot_enhanced_accuracy_lines_for_cifar(results_dir):

    # Impostazioni grafiche aggiornate per testi più grandi
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 20,  # dimensione di base del font
        "axes.titlesize": 26,  # dimensione del titolo
        "axes.labelsize": 22,  # dimensione delle etichette degli assi
    })

    pattern = os.path.join(results_dir, "*cifar0.1.json")
    json_files = glob.glob(pattern)

    if not json_files:
        print("Nessun file trovato con 'cifar1.0' nel nome.")
        return

    plt.figure(figsize=(12, 8))
    beta_value = None  # Per salvare il beta estratto dal primo file processato

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        if "accuracy_over_time" not in data:
            print(f"Il file {file_path} non contiene 'accuracy_over_time'.")
            continue

        peers_data = data["accuracy_over_time"]

        # Raccogli tutti i tempi presenti in tutti i peer per creare una timeline comune
        all_times = set()
        for peer in peers_data.values():
            all_times.update(peer.get("times", []))
        if not all_times:
            print(f"Il file {file_path} non contiene tempi validi.")
            continue
        common_times = np.array(sorted(all_times))

        # Estrai dataset e beta dal nome del file, atteso: "plot_values_<dataset><beta>.json"
        base = os.path.basename(file_path)
        core = base[len("plot_values_"):-len(".json")]
        match = re.match(r"([a-zA-Z]+)([0-9.]+)", core)
        if match:
            dataset = match.group(1)
            current_beta = match.group(2)
        else:
            dataset = core
            current_beta = ""
        if beta_value is None and current_beta:
            beta_value = current_beta

        # Per ciascun peer interpola e traccia la sua linea
        for peer_id, peer in peers_data.items():
            times = np.array(peer["times"])
            accuracies = np.array(peer["accuracies"])
            interp_acc = np.interp(common_times, times, accuracies)
            line, = plt.plot(common_times, interp_acc, linewidth=1.5, marker="o", markersize=3,
                             label=f"Peer {peer_id}")
            plt.fill_between(common_times, interp_acc, alpha=0.01, color=line.get_color())

    plt.title("Accuracy Over Time on $\mathit{cifar10}$", fontsize=26)
    plt.xlabel("Time (s)", fontsize=22)
    plt.ylabel("Accuracy (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc="lower right")

    # Aggiungi il valore di β al centro del plot (senza riquadro)
    if beta_value:
        plt.text(0.5, 0.5, f"β = {beta_value}", transform=plt.gca().transAxes,
                 fontsize=25, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='none'))

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    plot_enhanced_accuracy_lines_for_mnist("results")
    plot_accuracy_boxplot_by_dataset("results")
    plot_peer_flow_bar_with_progress("results")
    plot_enhanced_accuracy_lines_for_cifar("results")


if __name__ == "__main__":
    main()











