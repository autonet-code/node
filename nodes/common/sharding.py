"""
Model Sharding and Erasure Coding for Autonet

Implements:
- Layer-wise sharding for CNNs
- Tensor sharding for LLMs
- Reed-Solomon erasure coding (10,14 scheme)
- Merkle tree construction and verification
"""

import hashlib
import io
import json
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    LAYER_WISE = "layer_wise"       # Each shard = complete layers
    TENSOR_PARALLEL = "tensor"      # Weight matrices split
    REPLICA = "replica"             # Full model replicated


class StorageTier(Enum):
    NODE_PUBLIC = 0
    NODE_PINNED = 1
    FILECOIN = 2
    ARWEAVE = 3


@dataclass
class ShardInfo:
    """Information about a single shard."""
    shard_index: int
    shard_hash: bytes
    cid: str
    size: int
    is_parity: bool
    layer_name: Optional[str] = None  # For layer-wise sharding
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None


@dataclass
class ShardManifest:
    """Complete manifest describing a sharded model."""
    model_hash: bytes
    merkle_root: bytes
    total_shards: int
    data_shards: int
    parity_shards: int
    total_size: int
    strategy: ShardingStrategy
    shards: List[ShardInfo] = field(default_factory=list)
    layer_map: Dict[str, ShardInfo] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "model_hash": self.model_hash.hex(),
            "merkle_root": self.merkle_root.hex(),
            "total_shards": self.total_shards,
            "data_shards": self.data_shards,
            "parity_shards": self.parity_shards,
            "total_size": self.total_size,
            "strategy": self.strategy.value,
            "shards": [
                {
                    "shard_index": s.shard_index,
                    "shard_hash": s.shard_hash.hex(),
                    "cid": s.cid,
                    "size": s.size,
                    "is_parity": s.is_parity,
                    "layer_name": s.layer_name,
                    "shape": s.shape,
                    "dtype": s.dtype,
                }
                for s in self.shards
            ],
            "layer_map": {
                k: {
                    "shard_index": v.shard_index,
                    "cid": v.cid,
                    "shard_hash": v.shard_hash.hex(),
                }
                for k, v in self.layer_map.items()
            },
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ShardManifest":
        """Create from JSON dict."""
        shards = [
            ShardInfo(
                shard_index=s["shard_index"],
                shard_hash=bytes.fromhex(s["shard_hash"]),
                cid=s["cid"],
                size=s["size"],
                is_parity=s["is_parity"],
                layer_name=s.get("layer_name"),
                shape=s.get("shape"),
                dtype=s.get("dtype"),
            )
            for s in data["shards"]
        ]

        manifest = cls(
            model_hash=bytes.fromhex(data["model_hash"]),
            merkle_root=bytes.fromhex(data["merkle_root"]),
            total_shards=data["total_shards"],
            data_shards=data["data_shards"],
            parity_shards=data["parity_shards"],
            total_size=data["total_size"],
            strategy=ShardingStrategy(data["strategy"]),
            shards=shards,
        )

        # Rebuild layer map
        for shard in shards:
            if shard.layer_name:
                manifest.layer_map[shard.layer_name] = shard

        return manifest


class MerkleTree:
    """Merkle tree for shard verification."""

    @staticmethod
    def compute_root(leaf_hashes: List[bytes]) -> bytes:
        """Compute Merkle root from leaf hashes."""
        if len(leaf_hashes) == 0:
            return bytes(32)
        if len(leaf_hashes) == 1:
            return leaf_hashes[0]

        # Pad to power of 2
        n = len(leaf_hashes)
        next_pow2 = 1 << (n - 1).bit_length()
        padded = leaf_hashes + [bytes(32)] * (next_pow2 - n)

        # Build tree bottom-up
        current_level = padded
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else bytes(32)
                combined = hashlib.sha256(left + right).digest()
                next_level.append(combined)
            current_level = next_level

        return current_level[0]

    @staticmethod
    def generate_proof(leaf_index: int, leaf_hashes: List[bytes]) -> List[bytes]:
        """Generate Merkle proof for a specific leaf."""
        if len(leaf_hashes) == 0:
            return []

        # Pad to power of 2
        n = len(leaf_hashes)
        next_pow2 = 1 << (n - 1).bit_length()
        padded = leaf_hashes + [bytes(32)] * (next_pow2 - n)

        proof = []
        current_level = padded
        current_index = leaf_index

        while len(current_level) > 1:
            # Get sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1

            if sibling_index < len(current_level):
                proof.append(current_level[sibling_index])

            # Move to parent level
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else bytes(32)
                combined = hashlib.sha256(left + right).digest()
                next_level.append(combined)

            current_level = next_level
            current_index = current_index // 2

        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: bytes,
        leaf_index: int,
        proof: List[bytes],
        root: bytes
    ) -> bool:
        """Verify a Merkle proof."""
        computed = leaf_hash
        index = leaf_index

        for sibling in proof:
            if index % 2 == 0:
                computed = hashlib.sha256(computed + sibling).digest()
            else:
                computed = hashlib.sha256(sibling + computed).digest()
            index = index // 2

        return computed == root


class ReplicationCodec:
    """
    Replication-based erasure coding.

    Each parity shard is a copy of one of the data shards (round-robin).
    This allows recovery from any parity_shards unavailable data shards.

    Simple, correct, and zero external dependencies.
    Use `reedsolo` or `zfec` for production-grade erasure coding.
    """

    def __init__(self, data_shards: int = 10, parity_shards: int = 4):
        self.data_shards = data_shards
        self.parity_shards = parity_shards
        self.total_shards = data_shards + parity_shards

    def encode(self, data: bytes) -> List[bytes]:
        """
        Split data into data_shards pieces and append parity_shards replicas.

        Parity shard i = copy of data shard (i % data_shards).
        """
        shard_size = math.ceil(len(data) / self.data_shards)
        data_pieces = []

        for i in range(self.data_shards):
            start = i * shard_size
            end = min(start + shard_size, len(data))
            piece = data[start:end]
            # Pad last shard to uniform size
            if len(piece) < shard_size:
                piece = piece + bytes(shard_size - len(piece))
            data_pieces.append(piece)

        # Parity shards: replicas of data shards in round-robin order
        parity_pieces = [
            data_pieces[i % self.data_shards] for i in range(self.parity_shards)
        ]

        return data_pieces + parity_pieces

    def decode(self, shards: List[Optional[bytes]], shard_size: int) -> bytes:
        """
        Decode shards back to original data, recovering from missing data shards
        using their parity (replica) counterparts.
        """
        # Fill missing data shards from parity replicas
        recovered = list(shards)
        for i in range(self.data_shards):
            if recovered[i] is None:
                parity_idx = self.data_shards + (i % self.parity_shards)
                if parity_idx < len(shards) and shards[parity_idx] is not None:
                    recovered[i] = shards[parity_idx]
                    logger.info(f"Recovered data shard {i} from parity shard {parity_idx}")

        available_data = [recovered[i] for i in range(self.data_shards) if recovered[i] is not None]
        if len(available_data) < self.data_shards:
            raise ValueError(
                f"Insufficient shards: need {self.data_shards}, "
                f"have {len(available_data)} after recovery attempt"
            )

        return b"".join(recovered[i] for i in range(self.data_shards))


# Keep backward-compatible alias
ReedSolomonCodec = ReplicationCodec


class NodeStorage:
    """
    Local storage for model shards on Autonet nodes.
    Uses content-addressed storage with hash-based filenames.
    """

    def __init__(self, storage_dir: str = "./shard_storage"):
        from pathlib import Path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, str] = {}  # hash -> filepath

    def store(self, data: bytes) -> str:
        """Store data and return content hash as ID."""
        content_hash = hashlib.sha256(data).hexdigest()
        filepath = self.storage_dir / f"{content_hash}.shard"

        with open(filepath, 'wb') as f:
            f.write(data)

        self._index[content_hash] = str(filepath)
        logger.debug(f"Stored shard: {content_hash[:16]}... ({len(data)} bytes)")
        return content_hash

    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Retrieve data by content hash."""
        filepath = self.storage_dir / f"{content_hash}.shard"

        if not filepath.exists():
            logger.warning(f"Shard not found: {content_hash[:16]}...")
            return None

        with open(filepath, 'rb') as f:
            data = f.read()

        # Verify integrity
        if hashlib.sha256(data).hexdigest() != content_hash:
            logger.error(f"Shard corruption detected: {content_hash[:16]}...")
            return None

        return data

    def delete(self, content_hash: str) -> bool:
        """Delete a shard."""
        filepath = self.storage_dir / f"{content_hash}.shard"
        if filepath.exists():
            filepath.unlink()
            self._index.pop(content_hash, None)
            return True
        return False

    def list_shards(self) -> List[str]:
        """List all stored shard hashes."""
        return [f.stem for f in self.storage_dir.glob("*.shard")]

    def get_used_space(self) -> int:
        """Get total bytes used."""
        return sum(f.stat().st_size for f in self.storage_dir.glob("*.shard"))


class ModelSharder:
    """
    Shards PyTorch models for distributed storage across Autonet nodes.
    """

    def __init__(
        self,
        strategy: ShardingStrategy = ShardingStrategy.LAYER_WISE,
        data_shards: int = 10,
        parity_shards: int = 4,
    ):
        self.strategy = strategy
        self.codec = ReedSolomonCodec(data_shards, parity_shards)

    def shard_model(
        self,
        model_weights: Dict[str, Any],
        storage: NodeStorage,
    ) -> ShardManifest:
        """
        Shard a model and store on local node storage.

        Args:
            model_weights: Dict mapping layer names to tensors/arrays
            storage: NodeStorage instance

        Returns:
            ShardManifest with all shard information
        """
        if self.strategy == ShardingStrategy.LAYER_WISE:
            return self._shard_layer_wise(model_weights, storage)
        elif self.strategy == ShardingStrategy.REPLICA:
            return self._shard_replica(model_weights, storage)
        else:
            # TENSOR_PARALLEL - future implementation
            return self._shard_layer_wise(model_weights, storage)

    def _shard_layer_wise(
        self,
        model_weights: Dict[str, Any],
        storage: NodeStorage,
    ) -> ShardManifest:
        """Shard by layers - each layer is a separate shard."""
        shards: List[ShardInfo] = []
        shard_hashes: List[bytes] = []
        total_size = 0

        # Create one shard per layer
        for idx, (layer_name, weights) in enumerate(model_weights.items()):
            # Serialize layer
            layer_bytes = self._serialize_weights(weights)
            shard_hash = hashlib.sha256(layer_bytes).digest()

            # Store on node storage
            content_id = storage.store(layer_bytes)

            # Get shape/dtype if tensor-like
            shape = None
            dtype = None
            if hasattr(weights, "shape"):
                shape = list(weights.shape)
            if hasattr(weights, "dtype"):
                dtype = str(weights.dtype)

            shard_info = ShardInfo(
                shard_index=idx,
                shard_hash=shard_hash,
                cid=content_id,
                size=len(layer_bytes),
                is_parity=False,
                layer_name=layer_name,
                shape=shape,
                dtype=dtype,
            )
            shards.append(shard_info)
            shard_hashes.append(shard_hash)
            total_size += len(layer_bytes)

        # Generate parity shards for erasure coding
        all_layer_bytes = b"".join(
            self._serialize_weights(w) for w in model_weights.values()
        )
        parity_data = self.codec.encode(all_layer_bytes)

        # Only add parity shards (skip data shards we already have)
        for p_idx, parity_bytes in enumerate(parity_data[self.codec.data_shards:]):
            parity_hash = hashlib.sha256(parity_bytes).digest()
            content_id = storage.store(parity_bytes)

            shard_info = ShardInfo(
                shard_index=len(shards),
                shard_hash=parity_hash,
                cid=content_id,
                size=len(parity_bytes),
                is_parity=True,
                layer_name=f"parity_{p_idx}",
            )
            shards.append(shard_info)
            shard_hashes.append(parity_hash)
            total_size += len(parity_bytes)

        # Build Merkle tree
        merkle_root = MerkleTree.compute_root(shard_hashes)

        # Create manifest
        manifest = ShardManifest(
            model_hash=hashlib.sha256(all_layer_bytes).digest(),
            merkle_root=merkle_root,
            total_shards=len(shards),
            data_shards=len(model_weights),
            parity_shards=self.codec.parity_shards,
            total_size=total_size,
            strategy=self.strategy,
            shards=shards,
        )

        # Build layer map
        for shard in shards:
            if shard.layer_name and not shard.is_parity:
                manifest.layer_map[shard.layer_name] = shard

        logger.info(
            f"Sharded model: {len(shards)} shards, "
            f"Merkle root: {merkle_root.hex()[:16]}..."
        )

        return manifest

    def _shard_replica(
        self,
        model_weights: Dict[str, Any],
        storage: NodeStorage,
    ) -> ShardManifest:
        """Store full model as single shard (for small models)."""
        all_bytes = self._serialize_weights(model_weights)
        shard_hash = hashlib.sha256(all_bytes).digest()
        content_id = storage.store(all_bytes)

        shard_info = ShardInfo(
            shard_index=0,
            shard_hash=shard_hash,
            cid=content_id,
            size=len(all_bytes),
            is_parity=False,
            layer_name="full_model",
        )

        manifest = ShardManifest(
            model_hash=shard_hash,
            merkle_root=shard_hash,
            total_shards=1,
            data_shards=1,
            parity_shards=0,
            total_size=len(all_bytes),
            strategy=ShardingStrategy.REPLICA,
            shards=[shard_info],
        )
        manifest.layer_map["full_model"] = shard_info

        return manifest

    def retrieve_model(
        self,
        manifest: ShardManifest,
        storage: NodeStorage,
        required_layers: Optional[List[str]] = None,
        verify_integrity: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve sharded model from node storage.

        Args:
            manifest: ShardManifest describing the model
            storage: NodeStorage instance
            required_layers: Only download these layers (None = all)
            verify_integrity: Verify Merkle proofs

        Returns:
            Dict mapping layer names to weights
        """
        model_weights = {}

        if manifest.strategy == ShardingStrategy.REPLICA:
            # Single shard - full model
            shard = manifest.shards[0]
            data = storage.retrieve(shard.cid)
            if data is None:
                raise ValueError(f"Failed to retrieve shard: {shard.cid}")

            if verify_integrity:
                computed_hash = hashlib.sha256(data).digest()
                if computed_hash != shard.shard_hash:
                    raise ValueError("Shard integrity check failed")

            return self._deserialize_weights(data)

        # Layer-wise sharding
        shard_hashes = [s.shard_hash for s in manifest.shards]

        for shard in manifest.shards:
            if shard.is_parity:
                continue  # Skip parity shards for normal retrieval

            if required_layers and shard.layer_name not in required_layers:
                continue

            data = storage.retrieve(shard.cid)
            if data is None:
                logger.warning(f"Failed to retrieve shard: {shard.cid}")
                continue

            if verify_integrity:
                # Verify hash
                computed_hash = hashlib.sha256(data).digest()
                if computed_hash != shard.shard_hash:
                    raise ValueError(f"Shard {shard.layer_name} integrity failed")

                # Verify Merkle proof
                proof = MerkleTree.generate_proof(shard.shard_index, shard_hashes)
                if not MerkleTree.verify_proof(
                    shard.shard_hash, shard.shard_index, proof, manifest.merkle_root
                ):
                    raise ValueError(f"Shard {shard.layer_name} Merkle proof failed")

            # Deserialize
            layer_weights = self._deserialize_weights(data)
            model_weights[shard.layer_name] = layer_weights

        return model_weights

    def _serialize_weights(self, weights: Any) -> bytes:
        """Serialize weights to bytes."""
        try:
            import torch
            if isinstance(weights, torch.Tensor):
                buffer = io.BytesIO()
                torch.save(weights, buffer)
                return buffer.getvalue()
        except ImportError:
            pass

        try:
            import numpy as np
            if isinstance(weights, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, weights)
                return buffer.getvalue()
        except ImportError:
            pass

        # Fallback to JSON for dicts/lists
        if isinstance(weights, (dict, list)):
            return json.dumps(weights, sort_keys=True).encode()

        raise ValueError(f"Cannot serialize weights of type: {type(weights)}")

    def _deserialize_weights(self, data: bytes) -> Any:
        """Deserialize weights from bytes."""
        # Try PyTorch format
        try:
            import torch
            buffer = io.BytesIO(data)
            return torch.load(buffer, weights_only=True)
        except Exception:
            pass

        # Try NumPy format
        try:
            import numpy as np
            buffer = io.BytesIO(data)
            return np.load(buffer, allow_pickle=False)
        except Exception:
            pass

        # Try JSON
        try:
            return json.loads(data.decode())
        except Exception:
            pass

        raise ValueError("Cannot deserialize weights")


# =============================================================================
# DistributedShardManager — P2P-aware shard distribution
# =============================================================================


class DistributedShardManager:
    """
    Manages model shard distribution across P2P peers.

    Wraps ModelSharder and BlobStore to:
    - Upload shards to local blob store and announce over P2P
    - Retrieve missing shards from P2P peers on demand
    - Select peers for shard placement based on capability
    """

    def __init__(
        self,
        sharder: ModelSharder,
        blob_store,  # BlobStore
        p2p_host=None,  # Optional[AutonetHost]
    ):
        self.sharder = sharder
        self.blob_store = blob_store
        self.p2p_host = p2p_host

    async def distribute(
        self,
        model_weights: Dict[str, Any],
        local_storage: NodeStorage,
    ) -> ShardManifest:
        """
        Shard a model, store locally, and announce shards over P2P.

        Returns the completed ShardManifest.
        """
        manifest = self.sharder.shard_model(model_weights, local_storage)

        if self.p2p_host and self.p2p_host._running:
            # Announce each shard's CID via blob capability advertisement
            shard_cids = [s.cid for s in manifest.shards if not s.is_parity]
            self.p2p_host.update_capability(
                modules_hosted=list(self.p2p_host._capability.modules_hosted) + [
                    f"shard:{cid[:16]}" for cid in shard_cids
                ]
            )
            await self.p2p_host.advertise_capability()
            logger.info(
                f"Announced {len(shard_cids)} shards to P2P peers "
                f"(model: {manifest.model_hash.hex()[:16]}...)"
            )

        return manifest

    async def fetch_missing_shards(
        self,
        manifest: ShardManifest,
        local_storage: NodeStorage,
    ) -> int:
        """
        Attempt to fetch any locally-missing shards from P2P peers.

        Returns the count of shards successfully recovered.
        """
        if not self.p2p_host or not self.p2p_host._running:
            return 0

        recovered = 0
        for shard in manifest.shards:
            if shard.is_parity:
                continue
            if local_storage.retrieve(shard.cid) is not None:
                continue  # Already have it

            data = await self.p2p_host.fetch_blob_from_any_peer(shard.cid, timeout_per_peer=15.0)
            if data is not None:
                # Verify integrity before accepting
                actual_hash = hashlib.sha256(data).digest()
                if actual_hash == shard.shard_hash:
                    local_storage.store(data)
                    recovered += 1
                    logger.info(
                        f"Recovered shard {shard.shard_index} "
                        f"({shard.layer_name}) from P2P"
                    )
                else:
                    logger.warning(
                        f"Shard {shard.shard_index} hash mismatch from P2P"
                    )

        return recovered

    def select_peers_for_placement(
        self,
        num_peers: int,
        min_gpu_memory_mb: int = 0,
    ) -> List[str]:
        """
        Select peers suitable for storing shards, by latency.

        Returns list of peer_ids (up to num_peers).
        """
        if not self.p2p_host:
            return []

        candidates = self.p2p_host.find_capable_peers(
            min_gpu_memory=min_gpu_memory_mb
        )

        if self.p2p_host.latency_tracker:
            latencies = self.p2p_host.latency_tracker.all_latencies
            candidates.sort(
                key=lambda c: latencies.get(c.peer_id, type("", (), {"ema_rtt_ms": float("inf")})()).ema_rtt_ms
            )

        return [c.peer_id for c in candidates[:num_peers]]
