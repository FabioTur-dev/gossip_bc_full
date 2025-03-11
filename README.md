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
   git clone https://github.com/FabioTur-dev/gossip_bc_full.git

# Ethereum Development Setup Guide for Windows

This guide provides step-by-step instructions to install Node.js (with npm), Truffle, and Ganache CLI on a Windows system. It also includes brief explanations of what Truffle and Ganache are and how they are used in Ethereum development.

## Overview

- **Node.js:** A JavaScript runtime built on Chrome's V8 engine that allows you to run JavaScript on the server-side.
- **npm:** The package manager for Node.js, used for installing and managing libraries and tools.
- **Truffle:** An Ethereum development framework that simplifies the process of writing, testing, and deploying smart contracts.
- **Ganache CLI:** A command-line blockchain emulator for Ethereum that provides a local blockchain environment for rapid development and testing of smart contracts.

## 1. Install Node.js and npm

### Steps to Install on Windows:
1. Visit the official website: [nodejs.org](https://nodejs.org/).
2. Download the Windows installer (a `.msi` file).
3. Run the installer and follow the on-screen instructions. The installer includes npm by default.

### Verify the Installation
Open **Command Prompt** or **PowerShell** and run the following commands:

```bash
node -v
npm -v
```

### Install Truffle
```bash
npm install -g truffle
```

### Install Ganache
```bash
npm install -g ganache-cli
```

# IPFS (Kubo Version) Installation Guide for Windows

This guide explains what IPFS is, why the Kubo version is recommended, and provides step-by-step instructions to install it on Windows.

## What is IPFS?

IPFS (InterPlanetary File System) is a peer-to-peer distributed file system that enables decentralized storage and sharing of data. It connects all computing devices with the same system of files, making it possible to build resilient and distributed applications.

## Why the Kubo Version?

The Kubo version (formerly known as go-ipfs) is the most widely used and production-ready implementation of IPFS. It is written in Go, which offers high performance, stability, and efficiency. This version is generally preferred for production environments over the JavaScript (web) version due to its maturity and robustness.

## Installation Steps for IPFS (Kubo Version) on Windows

1. **Download the Latest Release:**
   - Visit the [Kubo GitHub Releases page](https://github.com/ipfs/kubo/releases).
   - Download the latest Windows release (usually provided as a `.zip` file).

2. **Extract and Install:**
   - Extract the downloaded archive to a directory of your choice.
   - Add the directory to your system's PATH environment variable to allow running the `ipfs` command from any terminal window.

3. **Verify the Installation:**
   - Open **Command Prompt** or **PowerShell**.
   - Run the following command to check the installed version:

```bash
ipfs version
```

# After Installation Guide

This guide explains how to initialize and start all the necessary components after installation, including Ganache, Truffle, and IPFS.

## Starting Ganache

To start your local blockchain using Ganache, run the following command in your terminal:

```bash
ganache --networkId 1337
```

This command starts Ganache with the specified network ID (1337).

## Initializing and Using Truffle

Truffle is used to develop, compile, deploy, and test your smart contracts. Below are common Truffle commands explained in English for use in PowerShell:

### 1. Initialize a New Truffle Project

This command sets up a new Truffle project by creating the necessary project structure (Optional).

```bash
truffle init
```

### 2. Compile Smart Contracts

This command compiles all your smart contracts, with the `--reset` flag ensuring that any previous compilations are cleared (Mandatory beacuse the blockchain will reset).

```bash
truffle compile --reset
```

### 3. Launch the Truffle Console

This command opens an interactive console connected to your development network, allowing you to interact with your deployed contracts (Optional).

```bash
truffle console --network development
```

### 4. Deploy (Migrate) Smart Contracts

This command deploys your smart contracts to the development network. The `--reset` flag forces a redeployment (*see the compile section).

```bash
truffle migrate --network development --reset
```

### 5. Run Smart Contract Tests

This command executes all the tests defined for your smart contracts.

```bash
truffle test
```

## Restarting Ganache with Custom Settings

If needed (but suggest not to doing it), you can restart Ganache with a higher gas limit using:

```bash
ganache --gasLimit 10000000000 --networkId 1337
```

This command starts Ganache with a custom gas limit to accommodate more complex transactions.

## Starting IPFS

To run IPFS from PowerShell, follow these steps:

1. **Initialize IPFS:**

   Run the following command to set up IPFS:

   ```bash
   ipfs
   ```

2. **Start the IPFS Daemon:**

   Then, start the IPFS daemon with:

   ```bash
   ipfs daemon
   ```

The IPFS daemon starts the service that allows you to interact with the distributed file system. Once the daemon is running, open your web browser and navigate to [http://127.0.0.1:5001/](http://127.0.0.1:5001/) to access the IPFS web interface.

# Post-Migration Update Instructions

After running `truffle migrate` to deploy your contracts, you must update the following two variables in both `main_cluster.py` and `blockchain_functions_cluster.py` files with the new deployed contract addresses.

The variables to update are:

```python
EXT_CONTRACT_ADDRESS = "0x5843b4cbD4Bbb80F4cF6ee2d6812D42ff833B378"  # ExtendedHashStorage
FED_CLUSTERING_ADDRESS = "0xbeB84916920ED1E09d8F4a1CFdd3118cBD4dB164"  # FedClustering
```

## Conclusion

By following these steps, you will have a fully initialized development environment with Ganache, Truffle, and IPFS up and running. You are now ready to develop, test, and deploy your smart contracts and decentralized applications.
```






