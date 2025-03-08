# FedBGS: Privacy-Preserving Segmented Gossip Learning System

FedBGS is a decentralized gossip learning system that leverages blockchain technology, differential privacy (DP), and partial homomorphic encryption (HE) for secure and privacy-preserving model aggregation in federated learning settings. This repository provides a complete solution for deploying FedBGS using Ganache, IPFS, Truffle, and related tools.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Deployment and Setup](#deployment-and-setup)
  - [1. Ganache Setup](#1-ganache-setup)
  - [2. IPFS Setup](#2-ipfs-setup)
  - [3. Truffle and Smart Contract Deployment](#3-truffle-and-smart-contract-deployment)
- [Running FedBGS](#running-fedbgs)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- **Privacy Preservation:** Applies differential privacy (DP) to clip and perturb client gradients before aggregation.
- **Secure Aggregation:** Utilizes partial homomorphic encryption (HE) (via the Paillier scheme) to securely aggregate label distributions and model updates on-chain.
- **Decentralized Clustering:** Segments clients using federated k-means++ based on their local label distributions.
- **Hybrid Storage:** Stores model updates on IPFS and verifies update hashes on the blockchain.
- **Robust Aggregation:** Implements trimmed mean aggregation to mitigate the effects of outliers and adversarial updates.
- **Penalty System:** Validates update hashes from client to client via blockchain and IPFS, and penalizes clients (by token deduction) if validation fails.

## Requirements
- **Node.js** (v12+)
- **npm** (v6+)
- **Ganache CLI** or **Ganache GUI**
- **Truffle** (v5+)
- **IPFS** (v0.4+)
- **Git**

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/FedBGS.git
   cd FedBGS
