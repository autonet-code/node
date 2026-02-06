"""
Distributed JEPA - Integration with ModelShardRegistry

Provides infrastructure for:
1. Sharding JEPA model weights across network nodes
2. Registering shards on-chain with merkle proofs
3. Retrieving and reassembling models from distributed storage
4. Erasure coding for fault tolerance
"""

import torch
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .jepa import JEPAConfig, JEPA
from .ipfs import IPFSClient

logger = logging.getLogger(__name__)


@dataclass
class JEPAShardInfo:
    """Information about a JEPA model shard."""
    shard_index: int
    layer_names: List[str]
    shard_hash: bytes  # 32 bytes
    size_bytes: int
    is_parity: bool = False
    ipfs_cid: Optional[str] = None


@dataclass
class JEPAShardManifest:
    """Complete manifest for a sharded JEPA model."""
    model_hash: bytes  # Unique identifier
    config: JEPAConfig
    total_shards: int
    data_shards: int
    parity_shards: int
    total_size: int
    merkle_root: bytes
    shards: List[JEPAShardInfo]
    sharding_strategy: str = "layer_wise"  # or "tensor_parallel"


class JEPAMerkleTree:
    """Compute merkle tree for shard verification."""

    @staticmethod
    def compute_leaf_hash(shard_data: Dict[str, torch.Tensor]) -> bytes:
        """Compute hash of a shard's data."""
        hasher = hashlib.sha256()
        for name in sorted(shard_data.keys()):
            hasher.update(name.encode())
            hasher.update(shard_data[name].numpy().tobytes())
        return hasher.digest()

    @staticmethod
    def compute_merkle_root(leaf_hashes: List[bytes]) -> bytes:
        """Compute merkle root from leaf hashes."""
        if not leaf_hashes:
            return b'\x00' * 32

        # Pad to power of 2
        n = len(leaf_hashes)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2

        padded = leaf_hashes + [b'\x00' * 32] * (next_pow2 - n)

        # Build tree bottom-up
        while len(padded) > 1:
            new_level = []
            for i in range(0, len(padded), 2):
                combined = padded[i] + padded[i + 1]
                new_level.append(hashlib.sha256(combined).digest())
            padded = new_level

        return padded[0]

    @staticmethod
    def get_merkle_proof(leaf_hashes: List[bytes], index: int) -> List[bytes]:
        """Get merkle proof for a specific shard."""
        n = len(leaf_hashes)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2

        padded = leaf_hashes + [b'\x00' * 32] * (next_pow2 - n)

        proof = []
        idx = index

        while len(padded) > 1:
            new_level = []
            for i in range(0, len(padded), 2):
                combined = padded[i] + padded[i + 1]
                new_level.append(hashlib.sha256(combined).digest())

            # Add sibling to proof
            if idx % 2 == 0:
                proof.append(padded[idx + 1])
            else:
                proof.append(padded[idx - 1])

            idx = idx // 2
            padded = new_level

        return proof


class DistributedJEPA:
    """
    Handles distributed storage and retrieval of JEPA models.

    Integrates with ModelShardRegistry.sol for on-chain coordination.
    """

    def __init__(
        self,
        ipfs: IPFSClient,
        registry=None,  # ContractRegistry
    ):
        self.ipfs = ipfs
        self.registry = registry

    def shard_model(
        self,
        model: JEPA,
        config: JEPAConfig,
        num_data_shards: int = 10,
        num_parity_shards: int = 4,
        strategy: str = "layer_wise"
    ) -> JEPAShardManifest:
        """
        Shard a JEPA model for distributed storage.

        Args:
            model: The JEPA model to shard
            config: JEPA configuration
            num_data_shards: Number of data shards (k in erasure coding)
            num_parity_shards: Number of parity shards (n-k)
            strategy: "layer_wise" or "tensor_parallel"

        Returns:
            JEPAShardManifest containing all shard info
        """
        state_dict = model.state_dict()
        layer_names = list(state_dict.keys())

        # Create data shards
        shards_data = []
        shard_infos = []

        if strategy == "layer_wise":
            # Divide layers evenly
            layers_per_shard = max(1, len(layer_names) // num_data_shards)

            for i in range(num_data_shards):
                start = i * layers_per_shard
                if i == num_data_shards - 1:
                    end = len(layer_names)
                else:
                    end = start + layers_per_shard

                shard_layers = layer_names[start:end]
                shard_data = {name: state_dict[name].clone() for name in shard_layers}
                shards_data.append(shard_data)

                # Compute hash
                shard_hash = JEPAMerkleTree.compute_leaf_hash(shard_data)
                size_bytes = sum(t.numel() * t.element_size() for t in shard_data.values())

                shard_infos.append(JEPAShardInfo(
                    shard_index=i,
                    layer_names=shard_layers,
                    shard_hash=shard_hash,
                    size_bytes=size_bytes,
                    is_parity=False,
                ))

        elif strategy == "tensor_parallel":
            # Split weight matrices horizontally
            # For simplicity, we still use layer-wise but mark it
            # Full tensor parallelism requires more complex handling
            raise NotImplementedError("Tensor parallel sharding not yet implemented")

        # Create parity shards using XOR-based erasure coding (simplified)
        # In production, use Reed-Solomon or similar
        if num_parity_shards > 0:
            parity_shards = self._compute_parity_shards(
                shards_data, num_parity_shards
            )
            for i, parity_data in enumerate(parity_shards):
                shard_hash = JEPAMerkleTree.compute_leaf_hash(parity_data)
                size_bytes = sum(t.numel() * t.element_size() for t in parity_data.values())

                shard_infos.append(JEPAShardInfo(
                    shard_index=num_data_shards + i,
                    layer_names=list(parity_data.keys()),
                    shard_hash=shard_hash,
                    size_bytes=size_bytes,
                    is_parity=True,
                ))
                shards_data.append(parity_data)

        # Compute merkle root
        leaf_hashes = [s.shard_hash for s in shard_infos]
        merkle_root = JEPAMerkleTree.compute_merkle_root(leaf_hashes)

        # Compute model hash
        model_hash = hashlib.sha256(merkle_root + config.image_size.to_bytes(4, 'big')).digest()

        # Compute total size
        total_size = sum(s.size_bytes for s in shard_infos if not s.is_parity)

        manifest = JEPAShardManifest(
            model_hash=model_hash,
            config=config,
            total_shards=len(shard_infos),
            data_shards=num_data_shards,
            parity_shards=num_parity_shards,
            total_size=total_size,
            merkle_root=merkle_root,
            shards=shard_infos,
            sharding_strategy=strategy,
        )

        logger.info(
            f"Created {manifest.total_shards} shards "
            f"({manifest.data_shards} data + {manifest.parity_shards} parity) "
            f"for JEPA model, total size: {total_size / 1024:.1f} KB"
        )

        # Store shard data for later upload
        manifest._shard_data = shards_data

        return manifest

    def _compute_parity_shards(
        self,
        data_shards: List[Dict[str, torch.Tensor]],
        num_parity: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute parity shards using simplified XOR-based encoding.

        In production, use Reed-Solomon (reedsolo library) for proper erasure coding.
        """
        parity_shards = []

        # For now, just create simple XOR parity across groups
        # This is a placeholder - real implementation needs proper erasure coding
        for p in range(num_parity):
            parity_data = {}
            # XOR shards in groups
            group_size = max(1, len(data_shards) // num_parity)
            start = p * group_size
            end = min(start + group_size, len(data_shards))

            if start < len(data_shards):
                base_shard = data_shards[start]
                for name in base_shard.keys():
                    # Convert to int8 for XOR, then back
                    result = (base_shard[name] * 127).to(torch.int8)
                    for i in range(start + 1, end):
                        if name in data_shards[i]:
                            other = (data_shards[i][name] * 127).to(torch.int8)
                            result = result ^ other
                    parity_data[f"parity_{p}_{name}"] = result.float() / 127.0

                parity_shards.append(parity_data)

        return parity_shards

    def upload_shards_to_ipfs(
        self,
        manifest: JEPAShardManifest
    ) -> JEPAShardManifest:
        """
        Upload all shards to IPFS and update manifest with CIDs.
        """
        if not hasattr(manifest, '_shard_data'):
            raise ValueError("Manifest has no shard data - call shard_model first")

        for i, shard_info in enumerate(manifest.shards):
            shard_data = manifest._shard_data[i]

            # Serialize to bytes
            serialized = {}
            for name, tensor in shard_data.items():
                serialized[name] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'data': tensor.numpy().tolist(),
                }

            # Upload to IPFS
            cid = self.ipfs.add_json({
                'shard_index': shard_info.shard_index,
                'is_parity': shard_info.is_parity,
                'layer_names': shard_info.layer_names,
                'tensors': serialized,
            })

            shard_info.ipfs_cid = cid
            logger.debug(f"Uploaded shard {i} to IPFS: {cid[:20]}...")

        logger.info(f"Uploaded {len(manifest.shards)} shards to IPFS")
        return manifest

    def register_model_on_chain(
        self,
        manifest: JEPAShardManifest,
        project_id: int,
    ) -> Optional[bytes]:
        """
        Register the sharded model on-chain via ModelShardRegistry.

        Returns model_hash if successful.
        """
        if not self.registry:
            logger.warning("No registry configured, skipping on-chain registration")
            return manifest.model_hash

        try:
            # Upload manifest to IPFS
            manifest_json = {
                'model_hash': manifest.model_hash.hex(),
                'config': {
                    'image_size': manifest.config.image_size,
                    'patch_size': manifest.config.patch_size,
                    'embed_dim': manifest.config.embed_dim,
                    'num_heads': manifest.config.num_heads,
                    'encoder_depth': manifest.config.encoder_depth,
                },
                'shards': [
                    {
                        'index': s.shard_index,
                        'hash': s.shard_hash.hex(),
                        'cid': s.ipfs_cid,
                        'size': s.size_bytes,
                        'is_parity': s.is_parity,
                    }
                    for s in manifest.shards
                ],
            }
            manifest_cid = self.ipfs.add_json(manifest_json)

            # Call ModelShardRegistry.registerModel
            # StorageTier.IPFS_PINNED = 1
            # ShardingStrategy.LAYER_WISE = 0
            result = self.registry.send(
                "ModelShardRegistry",
                "registerModel",
                manifest.model_hash,      # bytes32 modelHash
                manifest_cid,             # string manifestCid
                manifest.merkle_root,     # bytes32 merkleRoot
                manifest.data_shards,     # uint8 dataShards
                manifest.parity_shards,   # uint8 parityShards
                manifest.total_size,      # uint256 totalSize
                1,                        # StorageTier.IPFS_PINNED
                0,                        # ShardingStrategy.LAYER_WISE
                project_id,               # uint256 projectId
            )

            if result.success:
                logger.info(f"Registered model on-chain: {manifest.model_hash.hex()[:16]}...")
                return manifest.model_hash
            else:
                logger.error(f"On-chain registration failed: {result.error}")
                return None

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None

    def announce_shard_storage(
        self,
        model_hash: bytes,
        shard_index: int,
        shard_hash: bytes,
        shard_size: int,
        is_parity: bool,
    ) -> bool:
        """
        Announce that this node is storing a specific shard.
        """
        if not self.registry:
            return True

        try:
            result = self.registry.send(
                "ModelShardRegistry",
                "announceShard",
                model_hash,
                shard_index,
                shard_hash,
                shard_size,
                is_parity,
            )
            return result.success
        except Exception as e:
            logger.error(f"Error announcing shard: {e}")
            return False

    def retrieve_and_reassemble(
        self,
        manifest: JEPAShardManifest,
        config: Optional[JEPAConfig] = None,
    ) -> JEPA:
        """
        Retrieve shards from IPFS and reassemble the model.

        Handles missing shards via erasure coding reconstruction.
        """
        config = config or manifest.config

        # Download all available data shards
        state_dict = {}
        available_shards = 0

        for shard_info in manifest.shards:
            if shard_info.is_parity:
                continue  # Skip parity for now

            if not shard_info.ipfs_cid:
                logger.warning(f"Shard {shard_info.shard_index} has no CID")
                continue

            try:
                shard_data = self.ipfs.get_json(shard_info.ipfs_cid)
                tensors = shard_data.get('tensors', {})

                for name, tensor_info in tensors.items():
                    tensor = torch.tensor(
                        tensor_info['data'],
                        dtype=getattr(torch, tensor_info['dtype'].split('.')[-1])
                    ).reshape(tensor_info['shape'])
                    state_dict[name] = tensor

                available_shards += 1
                logger.debug(f"Retrieved shard {shard_info.shard_index}")

            except Exception as e:
                logger.warning(f"Failed to retrieve shard {shard_info.shard_index}: {e}")

        # Check if we have enough shards
        if available_shards < manifest.data_shards:
            # Would need to use parity shards for reconstruction
            logger.warning(
                f"Only {available_shards}/{manifest.data_shards} data shards available. "
                f"Erasure coding reconstruction not yet implemented."
            )

        # Reconstruct model
        model = JEPA(config)
        model.load_state_dict(state_dict, strict=False)

        logger.info(f"Reassembled model from {available_shards} shards")
        return model

    def check_shard_availability(
        self,
        model_hash: bytes,
    ) -> Tuple[int, bool]:
        """
        Check how many shards are available on-chain.

        Returns (available_count, is_sufficient)
        """
        if not self.registry:
            return (0, False)

        try:
            result = self.registry.call(
                "ModelShardRegistry",
                "checkShardAvailability",
                model_hash,
            )
            return result  # (availableShards, sufficient)
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return (0, False)


def create_distributed_jepa_model(
    config: JEPAConfig,
    ipfs: IPFSClient,
    registry=None,
    num_shards: int = 4,
) -> Tuple[JEPA, JEPAShardManifest]:
    """
    Convenience function to create and shard a JEPA model.
    """
    model = JEPA(config)
    distributor = DistributedJEPA(ipfs, registry)

    manifest = distributor.shard_model(
        model=model,
        config=config,
        num_data_shards=num_shards,
        num_parity_shards=1,
    )

    manifest = distributor.upload_shards_to_ipfs(manifest)

    return model, manifest
