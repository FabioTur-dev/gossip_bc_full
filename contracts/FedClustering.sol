// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FedClustering {
    uint256 public dimension;
    uint256 public outputLayerSize;
    uint256[][] public centers;
    address public owner;

    // Mappatura peer -> segmento
    mapping(address => uint256) public peerSegment;
    // Lista dei peer registrati (per leader election)
    address[] public peers;

    constructor(uint256 _dimension, uint256 _numClusters) {
        owner = msg.sender;
        dimension = _dimension;
        for (uint256 i = 0; i < _numClusters; i++) {
            uint256[] memory center = new uint256[](_dimension);
            centers.push(center);
        }
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function.");
        _;
    }

    function setOutputLayerSize(uint256 _size) public onlyOwner {
        outputLayerSize = _size;
    }

    function setClusterCenter(uint256 clusterId, uint256[] memory newCenter) public onlyOwner {
        require(clusterId < centers.length, "Invalid clusterId.");
        require(newCenter.length == dimension, "Dimension mismatch.");
        centers[clusterId] = newCenter;
    }

    function getClusterCenter(uint256 clusterId) public view returns (uint256[] memory) {
        require(clusterId < centers.length, "Invalid clusterId.");
        return centers[clusterId];
    }

    function updateClusterCenter(uint256 clusterId, uint256[] memory newCenter) public onlyOwner {
        require(clusterId < centers.length, "Invalid clusterId.");
        require(newCenter.length == dimension, "Dimension mismatch.");
        centers[clusterId] = newCenter;
    }

    function getNumClusters() public view returns (uint256) {
        return centers.length;
    }

    // Assegna il segmento a un peer e lo registra (se non già presente)
    function assignPeerSegment(address _peer, uint256 segmentId) public onlyOwner {
        require(segmentId < centers.length, "Invalid segmentId.");
        peerSegment[_peer] = segmentId;
        bool exists = false;
        for (uint256 i = 0; i < peers.length; i++) {
            if (peers[i] == _peer) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            peers.push(_peer);
        }
    }

    // Restituisce il segmento assegnato a un peer
    function getPeerSegment(address _peer) public view returns (uint256) {
        return peerSegment[_peer];
    }

    // Calcola e restituisce le boundaries per il segmento specificato
    function getSegmentBoundaries(uint256 segmentId) public view returns (uint256, uint256) {
        require(outputLayerSize > 0, "Output layer size not set.");
        uint256 numClusters = centers.length;
        uint256 segmentSize = (outputLayerSize + numClusters - 1) / numClusters;
        require(segmentId < numClusters, "Invalid segmentId.");
        uint256 startIdx = segmentId * segmentSize;
        uint256 endIdx = startIdx + segmentSize;
        if (endIdx > outputLayerSize) {
            endIdx = outputLayerSize;
        }
        return (startIdx, endIdx);
    }

    // Leader election: restituisce un peer casuale dalla lista dei peer registrati
    function electLeader() public view returns (address) {
        require(peers.length > 0, "No peers registered");
        uint256 index = uint256(keccak256(abi.encodePacked(block.timestamp, block.difficulty))) % peers.length;
        return peers[index];
    }
}


