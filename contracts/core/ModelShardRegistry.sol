// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "./ParticipantStaking.sol";

/**
 * @title ModelShardRegistry
 * @dev Tracks which nodes hold which model shards for distributed storage and retrieval.
 *
 *      Enables:
 *      - Layer-wise sharding for CNNs (each node stores some layers)
 *      - Tensor sharding for LLMs (weight matrices split across nodes)
 *      - Erasure coding (10,14) for fault tolerance
 *      - Merkle proofs for shard verification
 *
 *      Storage providers stake ATN and earn rewards for storing shards.
 */
contract ModelShardRegistry is Ownable {
    ParticipantStaking public immutable staking;

    // Erasure coding configuration
    uint8 public constant DEFAULT_DATA_SHARDS = 10;
    uint8 public constant DEFAULT_PARITY_SHARDS = 4;
    uint256 public constant MIN_STORAGE_STAKE = 50 * 10**18; // 50 ATN

    enum StorageTier {
        IPFS_PUBLIC,      // Free, no guarantees
        IPFS_PINNED,      // Pinned by authorized nodes
        FILECOIN,         // Paid storage with replication
        ARWEAVE           // Permanent storage
    }

    enum ShardingStrategy {
        LAYER_WISE,       // For CNNs: each shard = complete layers
        TENSOR_PARALLEL,  // For LLMs: weight matrices split
        REPLICA           // Full model replicated across nodes
    }

    struct ModelShardManifest {
        string manifestCid;          // IPFS CID of shard manifest JSON
        bytes32 merkleRoot;          // Merkle root of all shard hashes
        uint8 totalShards;           // Total number of shards (data + parity)
        uint8 dataShards;            // Number of data shards (k in erasure code)
        uint8 parityShards;          // Number of parity shards (n-k)
        uint256 totalSize;           // Total model size in bytes
        uint256 createdAt;
        address creator;
        StorageTier tier;
        ShardingStrategy strategy;
        uint256 projectId;           // Associated Autonet project
    }

    struct ShardInfo {
        bytes32 shardHash;           // Hash of shard data
        uint8 shardIndex;            // Index in erasure coding scheme
        bool isParityShard;          // true if parity, false if data
        uint256 size;                // Shard size in bytes
        uint256 lastVerified;        // Last successful integrity check
    }

    struct StorageProvider {
        uint256 capacity;            // Total storage capacity in bytes
        uint256 used;                // Used storage in bytes
        uint256 reputation;          // 0-1000 score
        uint256 successfulVerifications;
        uint256 failedVerifications;
        bool active;
    }

    // Mappings
    mapping(bytes32 => ModelShardManifest) public modelManifests;
    mapping(bytes32 => mapping(uint8 => ShardInfo)) public shards;
    mapping(bytes32 => mapping(uint8 => address[])) public shardProviders;
    mapping(address => StorageProvider) public providers;
    mapping(address => bytes32[]) public providerModels;

    // Events
    event ModelRegistered(
        bytes32 indexed modelHash,
        string manifestCid,
        uint8 totalShards,
        uint256 projectId
    );
    event ShardAnnounced(
        bytes32 indexed modelHash,
        uint8 shardIndex,
        address indexed provider
    );
    event ShardVerified(
        bytes32 indexed modelHash,
        uint8 shardIndex,
        address indexed verifier,
        bool success
    );
    event ShardRemoved(
        bytes32 indexed modelHash,
        uint8 shardIndex,
        address indexed provider
    );
    event ProviderRegistered(address indexed provider, uint256 capacity);
    event ShardRecoveryTriggered(bytes32 indexed modelHash, uint8 shardIndex);

    constructor(address _staking, address _owner) Ownable(_owner) {
        staking = ParticipantStaking(_staking);
    }

    // ============ Provider Management ============

    /**
     * @notice Register as a storage provider with capacity.
     * @param capacityBytes Total storage capacity in bytes
     */
    function registerProvider(uint256 capacityBytes) external {
        require(capacityBytes > 0, "Capacity must be positive");
        require(
            staking.getStake(msg.sender).amount >= MIN_STORAGE_STAKE,
            "Insufficient stake"
        );

        providers[msg.sender] = StorageProvider({
            capacity: capacityBytes,
            used: 0,
            reputation: 500, // Start at 50%
            successfulVerifications: 0,
            failedVerifications: 0,
            active: true
        });

        emit ProviderRegistered(msg.sender, capacityBytes);
    }

    /**
     * @notice Update provider capacity.
     */
    function updateCapacity(uint256 newCapacity) external {
        require(providers[msg.sender].active, "Not a provider");
        require(newCapacity >= providers[msg.sender].used, "Cannot reduce below used");
        providers[msg.sender].capacity = newCapacity;
    }

    /**
     * @notice Deactivate as a provider.
     */
    function deactivateProvider() external {
        providers[msg.sender].active = false;
    }

    // ============ Model Registration ============

    /**
     * @notice Register a new sharded model with its manifest.
     * @param modelHash Unique hash identifying this model
     * @param manifestCid IPFS CID of the shard manifest JSON
     * @param merkleRoot Merkle root of all shard hashes
     * @param dataShards Number of data shards (k in erasure coding)
     * @param parityShards Number of parity shards (n-k in erasure coding)
     * @param totalSize Total model size in bytes
     * @param tier Storage tier for this model
     * @param strategy Sharding strategy used
     * @param projectId Associated Autonet project ID
     */
    function registerModel(
        bytes32 modelHash,
        string memory manifestCid,
        bytes32 merkleRoot,
        uint8 dataShards,
        uint8 parityShards,
        uint256 totalSize,
        StorageTier tier,
        ShardingStrategy strategy,
        uint256 projectId
    ) external {
        require(modelManifests[modelHash].createdAt == 0, "Model already registered");
        require(dataShards > 0, "Need at least 1 data shard");

        uint8 totalShards = dataShards + parityShards;

        modelManifests[modelHash] = ModelShardManifest({
            manifestCid: manifestCid,
            merkleRoot: merkleRoot,
            totalShards: totalShards,
            dataShards: dataShards,
            parityShards: parityShards,
            totalSize: totalSize,
            createdAt: block.timestamp,
            creator: msg.sender,
            tier: tier,
            strategy: strategy,
            projectId: projectId
        });

        emit ModelRegistered(modelHash, manifestCid, totalShards, projectId);
    }

    // ============ Shard Announcements ============

    /**
     * @notice Announce that a node is storing a specific shard.
     * @param modelHash Model identifier
     * @param shardIndex Shard index (0 to totalShards-1)
     * @param shardHash Hash of the shard data for verification
     * @param shardSize Size of this shard in bytes
     * @param isParityShard True if this is a parity shard
     */
    function announceShard(
        bytes32 modelHash,
        uint8 shardIndex,
        bytes32 shardHash,
        uint256 shardSize,
        bool isParityShard
    ) external {
        require(modelManifests[modelHash].createdAt > 0, "Model not registered");
        require(shardIndex < modelManifests[modelHash].totalShards, "Invalid shard index");
        require(providers[msg.sender].active, "Not an active provider");

        StorageProvider storage provider = providers[msg.sender];
        require(provider.used + shardSize <= provider.capacity, "Capacity exceeded");

        ShardInfo storage shard = shards[modelHash][shardIndex];

        // Initialize shard if first provider
        if (shard.lastVerified == 0) {
            shard.shardHash = shardHash;
            shard.shardIndex = shardIndex;
            shard.isParityShard = isParityShard;
            shard.size = shardSize;
            shard.lastVerified = block.timestamp;
        } else {
            // Verify hash matches existing announcements
            require(shard.shardHash == shardHash, "Shard hash mismatch");
        }

        // Check if provider already announced this shard
        address[] storage providerList = shardProviders[modelHash][shardIndex];
        for (uint256 i = 0; i < providerList.length; i++) {
            require(providerList[i] != msg.sender, "Already storing this shard");
        }

        // Add provider
        providerList.push(msg.sender);
        provider.used += shardSize;
        providerModels[msg.sender].push(modelHash);

        emit ShardAnnounced(modelHash, shardIndex, msg.sender);
    }

    /**
     * @notice Remove announcement (provider no longer storing shard).
     */
    function removeShard(bytes32 modelHash, uint8 shardIndex) external {
        address[] storage providerList = shardProviders[modelHash][shardIndex];

        for (uint256 i = 0; i < providerList.length; i++) {
            if (providerList[i] == msg.sender) {
                // Remove from list (swap and pop)
                providerList[i] = providerList[providerList.length - 1];
                providerList.pop();

                // Update provider storage
                providers[msg.sender].used -= shards[modelHash][shardIndex].size;

                emit ShardRemoved(modelHash, shardIndex, msg.sender);
                return;
            }
        }
        revert("Not storing this shard");
    }

    // ============ Verification ============

    /**
     * @notice Verify a shard's integrity using Merkle proof.
     * @param modelHash Model identifier
     * @param shardIndex Shard index
     * @param shardHash Hash of the shard data
     * @param merkleProof Merkle proof for this shard
     */
    function verifyShard(
        bytes32 modelHash,
        uint8 shardIndex,
        bytes32 shardHash,
        bytes32[] calldata merkleProof
    ) external returns (bool valid) {
        require(modelManifests[modelHash].createdAt > 0, "Model not registered");

        ShardInfo storage shard = shards[modelHash][shardIndex];

        // Verify hash matches registered shard
        if (shard.shardHash != shardHash) {
            emit ShardVerified(modelHash, shardIndex, msg.sender, false);
            return false;
        }

        // Verify Merkle proof
        bytes32 computedRoot = _computeMerkleRoot(shardHash, shardIndex, merkleProof);
        valid = computedRoot == modelManifests[modelHash].merkleRoot;

        if (valid) {
            shard.lastVerified = block.timestamp;
        }

        emit ShardVerified(modelHash, shardIndex, msg.sender, valid);
        return valid;
    }

    /**
     * @notice Report a provider failed verification (called by verifiers).
     */
    function reportVerificationFailure(
        bytes32 modelHash,
        uint8 shardIndex,
        address provider
    ) external {
        // In production, would require proof of failure
        // For now, just update reputation
        StorageProvider storage p = providers[provider];
        p.failedVerifications++;

        // Reduce reputation (min 0)
        if (p.reputation >= 50) {
            p.reputation -= 50;
        } else {
            p.reputation = 0;
        }
    }

    /**
     * @notice Report successful verification.
     */
    function reportVerificationSuccess(
        bytes32 modelHash,
        uint8 shardIndex,
        address provider
    ) external {
        StorageProvider storage p = providers[provider];
        p.successfulVerifications++;

        // Increase reputation (max 1000)
        if (p.reputation <= 990) {
            p.reputation += 10;
        } else {
            p.reputation = 1000;
        }
    }

    // ============ View Functions ============

    /**
     * @notice Check if a model has sufficient shards available for reconstruction.
     */
    function checkShardAvailability(bytes32 modelHash)
        external
        view
        returns (uint8 availableShards, bool sufficient)
    {
        ModelShardManifest memory manifest = modelManifests[modelHash];
        require(manifest.createdAt > 0, "Model not registered");

        availableShards = 0;
        for (uint8 i = 0; i < manifest.totalShards; i++) {
            if (shardProviders[modelHash][i].length > 0) {
                availableShards++;
            }
        }

        // Need at least k (dataShards) to reconstruct
        sufficient = availableShards >= manifest.dataShards;
    }

    /**
     * @notice Get list of providers for a specific shard.
     */
    function getShardProviders(bytes32 modelHash, uint8 shardIndex)
        external
        view
        returns (address[] memory)
    {
        return shardProviders[modelHash][shardIndex];
    }

    /**
     * @notice Get model manifest.
     */
    function getModelManifest(bytes32 modelHash)
        external
        view
        returns (
            string memory manifestCid,
            bytes32 merkleRoot,
            uint8 totalShards,
            uint8 dataShards,
            uint256 totalSize,
            StorageTier tier,
            ShardingStrategy strategy
        )
    {
        ModelShardManifest memory m = modelManifests[modelHash];
        return (
            m.manifestCid,
            m.merkleRoot,
            m.totalShards,
            m.dataShards,
            m.totalSize,
            m.tier,
            m.strategy
        );
    }

    /**
     * @notice Get provider info.
     */
    function getProviderInfo(address provider)
        external
        view
        returns (
            uint256 capacity,
            uint256 used,
            uint256 reputation,
            uint256 successfulVerifications,
            uint256 failedVerifications,
            bool active
        )
    {
        StorageProvider memory p = providers[provider];
        return (p.capacity, p.used, p.reputation, p.successfulVerifications, p.failedVerifications, p.active);
    }

    /**
     * @notice Trigger recovery for missing shard.
     */
    function triggerShardRecovery(bytes32 modelHash, uint8 shardIndex) external {
        require(modelManifests[modelHash].createdAt > 0, "Model not registered");
        require(shardProviders[modelHash][shardIndex].length == 0, "Shard not missing");

        // Check if we have enough shards for reconstruction
        (uint8 availableShards, bool sufficient) = this.checkShardAvailability(modelHash);
        require(sufficient, "Insufficient shards for recovery");

        emit ShardRecoveryTriggered(modelHash, shardIndex);
        // Off-chain nodes listen and perform recovery
    }

    // ============ Internal Functions ============

    /**
     * @notice Compute Merkle root from leaf hash and proof.
     */
    function _computeMerkleRoot(
        bytes32 leafHash,
        uint8 leafIndex,
        bytes32[] calldata proof
    ) internal pure returns (bytes32 root) {
        root = leafHash;
        uint256 index = leafIndex;

        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 sibling = proof[i];

            if (index % 2 == 0) {
                // Left node
                root = keccak256(abi.encodePacked(root, sibling));
            } else {
                // Right node
                root = keccak256(abi.encodePacked(sibling, root));
            }

            index = index / 2;
        }
    }
}
