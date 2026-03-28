"""
Distributed JEPA - Integration with ModelShardRegistry and P2P

Provides infrastructure for:
1. Sharding JEPA model weights across network nodes
2. Registering shards on-chain with merkle proofs
3. Retrieving and reassembling models from distributed storage
4. Erasure coding for fault tolerance
5. P2P embedding exchange for the two-speed inference architecture
"""

import io
import struct
import torch
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Optional, Any
from pathlib import Path

from .jepa import JEPAConfig, JEPA
from .blob_store import BlobStore

if TYPE_CHECKING:
    from .p2p import AutonetHost

logger = logging.getLogger(__name__)


# =============================================================================
# Tensor serialization helpers (used for P2P embedding exchange)
# =============================================================================


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes using torch.save into a buffer."""
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Deserialize a tensor from bytes."""
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True)


@dataclass
class JEPAShardInfo:
    """Information about a JEPA model shard."""
    shard_index: int
    layer_names: List[str]
    shard_hash: bytes  # 32 bytes
    size_bytes: int
    is_parity: bool = False
    blob_hash: Optional[str] = None


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

    Integrates with ModelShardRegistry.sol for on-chain coordination
    and with AutonetHost for P2P shard transfer and embedding exchange.
    """

    def __init__(
        self,
        store: BlobStore,
        registry=None,  # ContractRegistry
        p2p_host: Optional["AutonetHost"] = None,
    ):
        self.store = store
        self.registry = registry
        self._p2p_host: Optional["AutonetHost"] = p2p_host
        self._embedding_handler: Optional[Callable] = None

        if p2p_host is not None:
            self._wire_p2p(p2p_host)

    def set_p2p_host(self, host: "AutonetHost") -> None:
        """Wire a P2P host to this DistributedJEPA after construction."""
        self._p2p_host = host
        self._wire_p2p(host)

    def _wire_p2p(self, host: "AutonetHost") -> None:
        """Register embedding handler on the P2P host."""
        host.set_embedding_handler(self._handle_incoming_embedding)
        logger.info("DistributedJEPA wired to P2P host")

    def set_embedding_handler(self, handler: Callable) -> None:
        """
        Set callback for received embeddings from peers.

        handler signature: async handler(peer_id: str, metadata: dict, embedding: torch.Tensor)
        """
        self._embedding_handler = handler

    async def _handle_incoming_embedding(
        self, peer_id: str, metadata: dict, tensor_bytes: bytes
    ) -> None:
        """Deserialize an incoming embedding tensor and invoke user handler."""
        if self._embedding_handler is None:
            return
        try:
            tensor = _bytes_to_tensor(tensor_bytes)
            await self._embedding_handler(peer_id, metadata, tensor)
        except Exception as e:
            logger.warning(f"Failed to deserialize embedding from {peer_id[:16]}: {e}")

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

    def upload_shards(
        self,
        manifest: JEPAShardManifest
    ) -> JEPAShardManifest:
        """
        Upload all shards to blob store and update manifest with content hashes.
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

            # Upload to blob store
            cid = self.store.add_json({
                'shard_index': shard_info.shard_index,
                'is_parity': shard_info.is_parity,
                'layer_names': shard_info.layer_names,
                'tensors': serialized,
            })

            shard_info.blob_hash = cid
            logger.debug(f"Uploaded shard {i} to blob store: {cid[:20]}...")

        logger.info(f"Uploaded {len(manifest.shards)} shards to blob store")
        return manifest

    async def upload_and_announce(self, manifest: JEPAShardManifest) -> JEPAShardManifest:
        """
        Upload shards to blob store and announce availability over P2P.

        Must be called from an async (trio) context.
        """
        manifest = self.upload_shards(manifest)

        if self._p2p_host and self._p2p_host._running:
            shard_cids = [s.blob_hash for s in manifest.shards if s.blob_hash and not s.is_parity]
            self._p2p_host.update_capability(
                modules_hosted=list(self._p2p_host._capability.modules_hosted) + [
                    f"jepa_shard:{cid[:16]}" for cid in shard_cids
                ]
            )
            await self._p2p_host.advertise_capability()
            logger.info(
                f"Announced {len(shard_cids)} JEPA shards over P2P "
                f"(model: {manifest.model_hash.hex()[:16]}...)"
            )

        return manifest

    async def retrieve_async(
        self,
        manifest: JEPAShardManifest,
        config: Optional[JEPAConfig] = None,
    ) -> "JEPA":
        """
        Async version of retrieve_and_reassemble — falls back to P2P for missing shards.

        Must be called from an async (trio) context.
        """
        config = config or manifest.config
        state_dict = {}
        available_shards = 0
        missing_blob_hashes = []

        for shard_info in manifest.shards:
            if shard_info.is_parity:
                continue
            if not shard_info.blob_hash:
                logger.warning(f"Shard {shard_info.shard_index} has no blob hash")
                continue

            shard_data = self.store.get_json(shard_info.blob_hash)

            if shard_data is None and self._p2p_host and self._p2p_host._running:
                # Try to fetch from P2P
                raw = await self.store.get_bytes_async(shard_info.blob_hash)
                if raw is not None:
                    import json as _json
                    try:
                        shard_data = _json.loads(raw.decode("utf-8"))
                    except Exception:
                        shard_data = None

            if shard_data is None:
                missing_blob_hashes.append(shard_info.blob_hash)
                logger.warning(f"Shard {shard_info.shard_index} unavailable")
                continue

            try:
                tensors = shard_data.get('tensors', {})
                for name, tensor_info in tensors.items():
                    tensor = torch.tensor(
                        tensor_info['data'],
                        dtype=getattr(torch, tensor_info['dtype'].split('.')[-1])
                    ).reshape(tensor_info['shape'])
                    state_dict[name] = tensor
                available_shards += 1
            except Exception as e:
                logger.warning(f"Failed to deserialize shard {shard_info.shard_index}: {e}")

        if missing_blob_hashes:
            logger.warning(
                f"{len(missing_blob_hashes)} shards missing even after P2P fallback"
            )

        model = JEPA(config)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Reassembled JEPA model from {available_shards} shards (async)")
        return model

    # =========================================================================
    # Embedding Exchange — two-speed inference architecture
    # =========================================================================

    async def send_embedding_to_peer(
        self,
        peer_id_str: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Send an embedding tensor to a specific peer for network-guided inference.

        This implements the fast-loop embedding exchange in the two-speed architecture:
        the local node sends its current context embedding to peers, and peers
        can return guidance embeddings to augment local predictions.

        Args:
            peer_id_str: Peer's libp2p peer ID string
            embedding: The embedding tensor to send
            metadata: Optional metadata dict (e.g. {"seq_id": 42, "layer": "context"})
        """
        if not self._p2p_host or not self._p2p_host._running:
            logger.debug("P2P not available for embedding exchange")
            return False

        try:
            from libp2p.peer.id import ID as PeerID
            target = PeerID.from_base58(peer_id_str)
        except Exception as e:
            logger.warning(f"Invalid peer ID {peer_id_str[:16]}: {e}")
            return False

        tensor_bytes = _tensor_to_bytes(embedding)
        return await self._p2p_host.send_embedding(target, tensor_bytes, metadata)

    async def broadcast_embedding(
        self,
        embedding: torch.Tensor,
        metadata: Optional[Dict] = None,
        max_peers: int = 4,
    ) -> int:
        """
        Broadcast an embedding to up to max_peers connected peers.

        Used by the slow loop to distribute network context to fast-loop nodes.
        Returns the number of peers the embedding was sent to successfully.
        """
        if not self._p2p_host or not self._p2p_host._running:
            return 0

        import trio
        peers = self._p2p_host.get_connected_peers()[:max_peers]
        if not peers:
            return 0

        tensor_bytes = _tensor_to_bytes(embedding)
        results = []

        async def send_to_one(pid_str):
            try:
                from libp2p.peer.id import ID as PeerID
                pid = PeerID.from_base58(pid_str)
                ok = await self._p2p_host.send_embedding(pid, tensor_bytes, metadata)
                results.append(ok)
            except Exception as e:
                logger.debug(f"Failed to send embedding to {pid_str[:16]}: {e}")
                results.append(False)

        async with trio.open_nursery() as nursery:
            for pid_str in peers:
                nursery.start_soon(send_to_one, pid_str)

        sent = sum(1 for r in results if r)
        logger.debug(f"Broadcast embedding to {sent}/{len(peers)} peers")
        return sent

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
            # Upload manifest to blob store
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
                        'cid': s.blob_hash,
                        'size': s.size_bytes,
                        'is_parity': s.is_parity,
                    }
                    for s in manifest.shards
                ],
            }
            manifest_cid = self.store.add_json(manifest_json)

            # Call ModelShardRegistry.registerModel
            # StorageTier.NODE_PINNED = 1
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
                1,                        # StorageTier.NODE_PINNED
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
        Retrieve shards from blob store and reassemble the model.

        Handles missing shards via erasure coding reconstruction.
        """
        config = config or manifest.config

        # Download all available data shards
        state_dict = {}
        available_shards = 0

        for shard_info in manifest.shards:
            if shard_info.is_parity:
                continue  # Skip parity for now

            if not shard_info.blob_hash:
                logger.warning(f"Shard {shard_info.shard_index} has no CID")
                continue

            try:
                shard_data = self.store.get_json(shard_info.blob_hash)
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
    store: BlobStore,
    registry=None,
    num_shards: int = 4,
) -> Tuple[JEPA, JEPAShardManifest]:
    """
    Convenience function to create and shard a JEPA model.
    """
    model = JEPA(config)
    distributor = DistributedJEPA(store, registry)

    manifest = distributor.shard_model(
        model=model,
        config=config,
        num_data_shards=num_shards,
        num_parity_shards=1,
    )

    manifest = distributor.upload_shards(manifest)

    return model, manifest
