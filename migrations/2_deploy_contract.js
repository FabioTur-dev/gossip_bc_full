const FedClustering = artifacts.require("FedClustering");

module.exports = function (deployer) {
  // Imposta la dimensione del centro (es. numero di classi, ad esempio 10 per CIFAR10)
  const dimension = 10;
  // Imposta il numero di cluster (segmenti) desiderato, ad esempio 2
  const numClusters = 2;
  deployer.deploy(FedClustering, dimension, numClusters);
};
