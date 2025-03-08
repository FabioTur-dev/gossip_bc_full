const ExtendedHashStorage = artifacts.require("ExtendedHashStorage");

module.exports = function (deployer) {
  deployer.deploy(ExtendedHashStorage);
};
