// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ExtendedHashStorage {
    mapping(string => bool) private existingHashes;
    mapping(address => string) private registeredClients;
    mapping(address => uint256) private tokenBalance;
    address[] private clientList;
    address private admin;

    event HashSaved(string indexed hash);
    event ClientRegistered(address indexed client, string name);
    event TokensPenalized(address indexed penalized, uint256 amount);
    event TokensReset();

    constructor() {
        admin = msg.sender;
        registeredClients[admin] = "admin";
        clientList.push(admin);
        tokenBalance[admin] = 1000;
    }

    function saveHash(string memory hash) public onlyRegistered {
        require(!existingHashes[hash], "Hash already saved.");
        existingHashes[hash] = true;
        emit HashSaved(hash);
    }

    function checkIfHashExists(string memory hash) public view onlyRegistered returns (bool) {
        return existingHashes[hash];
    }

    function registerClient(string memory username, string memory password) public {
        require(keccak256(abi.encodePacked(username)) == keccak256(abi.encodePacked("admin")), "Invalid username");
        require(keccak256(abi.encodePacked(password)) == keccak256(abi.encodePacked("admin")), "Invalid password");
        require(bytes(registeredClients[msg.sender]).length == 0, "Client already registered.");
        registeredClients[msg.sender] = username;
        tokenBalance[msg.sender] = 100;
        clientList.push(msg.sender);
        emit ClientRegistered(msg.sender, username);
    }

    function getClientName(address client) public view returns (string memory) {
        return registeredClients[client];
    }

    function getRegisteredClients() public view returns (address[] memory) {
        return clientList;
    }

    function getTokenBalance(address client) public view returns (uint256) {
        return tokenBalance[client];
    }

    function penalizeClient(address client, uint256 amount) public onlyRegistered {
        require(client != msg.sender, "Cannot penalize yourself");
        require(tokenBalance[client] >= amount, "Insufficient tokens to penalize");
        tokenBalance[client] -= amount;
        emit TokensPenalized(client, amount);
    }

    function resetTokenBalances() public onlyAdmin {
        for (uint i = 0; i < clientList.length; i++){
            address client = clientList[i];
            if(client == admin) {
                tokenBalance[client] = 1000;
            } else {
                tokenBalance[client] = 100;
            }
        }
        emit TokensReset();
    }

    modifier onlyRegistered() {
        require(bytes(registeredClients[msg.sender]).length != 0, "Not a registered client");
        _;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
}

