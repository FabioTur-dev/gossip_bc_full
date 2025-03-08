// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FedClustering {
    // Dimensione del vettore centro (ad es. numero di classi)
    uint256 public dimension;
    // Array di centri: ogni centro è un vettore di uint256
    uint256[][] public centers;
    // Proprietario del contratto (solo lui può aggiornare i centri)
    address public owner;

    constructor(uint256 _dimension, uint256 _numClusters) {
        owner = msg.sender;
        dimension = _dimension;
        // Inizializza i centri con vettori a zero
        for (uint256 i = 0; i < _numClusters; i++) {
            uint256[] memory center = new uint256[](_dimension);
            centers.push(center);
        }
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function.");
        _;
    }

    // Imposta il centro per un determinato cluster
    function setClusterCenter(uint256 clusterId, uint256[] memory newCenter) public onlyOwner {
        require(clusterId < centers.length, "Invalid clusterId.");
        require(newCenter.length == dimension, "Dimension mismatch.");
        centers[clusterId] = newCenter;
    }

    // Ritorna il centro di un cluster
    function getClusterCenter(uint256 clusterId) public view returns (uint256[] memory) {
        require(clusterId < centers.length, "Invalid clusterId.");
        return centers[clusterId];
    }

    // Aggiorna il centro di un cluster
    function updateClusterCenter(uint256 clusterId, uint256[] memory newCenter) public onlyOwner {
        require(clusterId < centers.length, "Invalid clusterId.");
        require(newCenter.length == dimension, "Dimension mismatch.");
        centers[clusterId] = newCenter;
    }

    // Ritorna il numero di cluster
    function getNumClusters() public view returns (uint256) {
        return centers.length;
    }
}
