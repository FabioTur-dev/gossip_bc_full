import matplotlib.pyplot as plt


def plot_accuracy(peers_accuracies):
    plt.figure(figsize=(10, 6))
    for peer_id, accuracies in peers_accuracies.items():
        rounds = range(1, len(accuracies) + 1)
        plt.plot(rounds, accuracies, marker='o', label=f'Peer {peer_id}')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Peer Test Accuracy per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(peers_losses):
    plt.figure(figsize=(10, 6))
    for peer_id, losses in peers_losses.items():
        rounds = range(1, len(losses) + 1)
        plt.plot(rounds, losses, marker='o', label=f'Peer {peer_id}')
    plt.xlabel('Round')
    plt.ylabel('Training Loss')
    plt.title('Peer Training Loss per Round')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_accuracy_bar(peers_accuracies):
    final_accuracies = {peer_id: acc[-1] for peer_id, acc in peers_accuracies.items() if acc}
    plt.figure(figsize=(10, 6))
    plt.bar(final_accuracies.keys(), final_accuracies.values(), color='skyblue')
    plt.xlabel('Peer')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Test Accuracy per Peer')
    plt.xticks(list(final_accuracies.keys()))
    plt.tight_layout()
    plt.show()


def plot_combined(peers_accuracies, peers_losses):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    for peer_id, accuracies in peers_accuracies.items():
        rounds = range(1, len(accuracies) + 1)
        axs[0].plot(rounds, accuracies, marker='o', label=f'Peer {peer_id}')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Test Accuracy (%)')
    axs[0].set_title('Peer Test Accuracy per Round')
    axs[0].legend()
    axs[0].grid(True)

    for peer_id, losses in peers_losses.items():
        rounds = range(1, len(losses) + 1)
        axs[1].plot(rounds, losses, marker='o', label=f'Peer {peer_id}')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Training Loss')
    axs[1].set_title('Peer Training Loss per Round')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
