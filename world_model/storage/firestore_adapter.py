"""
Firestore persistence adapter for WorldModel.

Schema:
    users/{userId}/
        world_model (document): { name, created_at, updated_at, agents, metadata }
        observations (subcollection): { content, source_id, timestamp, embedding, metadata }
        trees (subcollection): { root_value, description, root_node, score, depth }

Requires firebase-admin SDK:
    pip install firebase-admin
"""

from datetime import datetime
from typing import Optional
import json

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from ..models.observation import Observation, ObservationStore
from ..models.agent import AgentSet
from ..models.tree import Tree, TreeStore
from .world_model_store import WorldModel


class FirestoreAdapter:
    """
    Firestore persistence for WorldModel.

    Usage:
        # Initialize once with service account
        adapter = FirestoreAdapter.from_service_account('path/to/serviceAccount.json')

        # Or use default credentials (Cloud Run, etc.)
        adapter = FirestoreAdapter()

        # Save/load world models
        adapter.save_world_model(user_id, model)
        model = adapter.load_world_model(user_id)
    """

    def __init__(self, db=None):
        """
        Initialize with existing Firestore client or create one.

        Args:
            db: Existing firestore.Client, or None to use default app
        """
        if not FIREBASE_AVAILABLE:
            raise ImportError(
                "firebase-admin not installed. Run: pip install firebase-admin"
            )

        if db is not None:
            self.db = db
        else:
            # Use default app (must be initialized elsewhere)
            self.db = firestore.client()

    @classmethod
    def from_service_account(cls, cred_path: str, project_id: Optional[str] = None) -> "FirestoreAdapter":
        """
        Initialize Firebase from service account JSON file.

        Args:
            cred_path: Path to serviceAccountKey.json
            project_id: Optional project ID override
        """
        if not FIREBASE_AVAILABLE:
            raise ImportError(
                "firebase-admin not installed. Run: pip install firebase-admin"
            )

        # Initialize app if not already done
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            options = {"projectId": project_id} if project_id else None
            firebase_admin.initialize_app(cred, options)

        return cls(firestore.client())

    @classmethod
    def from_emulator(cls, project_id: str = "demo-project") -> "FirestoreAdapter":
        """
        Connect to Firestore emulator for local development.

        Start emulator: firebase emulators:start --only firestore
        """
        import os
        os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"

        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": project_id})

        return cls(firestore.client())

    # =========================================================================
    # World Model Operations
    # =========================================================================

    def save_world_model(self, user_id: str, model: WorldModel) -> None:
        """
        Save a complete WorldModel to Firestore.

        Args:
            user_id: User's unique identifier
            model: WorldModel to save
        """
        user_ref = self.db.collection("users").document(user_id)
        batch = self.db.batch()

        # Update timestamps
        model.updated_at = datetime.now()

        # Main document (agents + metadata)
        batch.set(user_ref.collection("world_model").document("profile"), {
            "name": model.name,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "agents": model.agents.to_dict(),
            "metadata": model.metadata,
        })

        batch.commit()

        # Observations (subcollection) - batch in groups of 500
        self._save_observations(user_ref, model.observations)

        # Trees (subcollection)
        self._save_trees(user_ref, model.trees)

    def _save_observations(self, user_ref, store: ObservationStore) -> None:
        """Save observations to subcollection in batches."""
        obs_ref = user_ref.collection("observations")
        observations = store.all()

        # Batch writes (Firestore limit is 500 per batch)
        for i in range(0, len(observations), 500):
            batch = self.db.batch()
            for obs in observations[i:i+500]:
                doc_ref = obs_ref.document(obs.id)
                batch.set(doc_ref, {
                    "content": obs.content,
                    "source_id": obs.source_id,
                    "timestamp": obs.timestamp,
                    "embedding": obs.embedding,
                    "metadata": obs.metadata,
                })
            batch.commit()

    def _save_trees(self, user_ref, store: TreeStore) -> None:
        """Save trees to subcollection."""
        trees_ref = user_ref.collection("trees")

        for tree in store.all():
            doc_ref = trees_ref.document(tree.id)
            doc_ref.set({
                "root_value": tree.root_value,
                "description": tree.description,
                "root_node": tree.root_node.to_dict() if tree.root_node else None,
                "score": tree.score,
                "depth": tree.depth(),
            })

    def load_world_model(self, user_id: str) -> Optional[WorldModel]:
        """
        Load a complete WorldModel from Firestore.

        Args:
            user_id: User's unique identifier

        Returns:
            WorldModel or None if not found
        """
        user_ref = self.db.collection("users").document(user_id)

        # Load main profile
        profile_doc = user_ref.collection("world_model").document("profile").get()
        if not profile_doc.exists:
            return None

        profile = profile_doc.to_dict()

        model = WorldModel(
            name=profile.get("name", ""),
            created_at=profile.get("created_at", datetime.now()),
            updated_at=profile.get("updated_at", datetime.now()),
            metadata=profile.get("metadata", {}),
        )

        # Load agents
        if "agents" in profile:
            model.agents = AgentSet.from_dict(profile["agents"])

        # Load observations
        model.observations = self._load_observations(user_ref)

        # Load trees
        model.trees = self._load_trees(user_ref)

        return model

    def _load_observations(self, user_ref) -> ObservationStore:
        """Load observations from subcollection."""
        store = ObservationStore()
        obs_docs = user_ref.collection("observations").stream()

        for doc in obs_docs:
            data = doc.to_dict()
            obs = Observation(
                id=doc.id,
                content=data.get("content", ""),
                source_id=data.get("source_id", ""),
                timestamp=data.get("timestamp", datetime.now()),
                embedding=data.get("embedding"),
                metadata=data.get("metadata", {}),
            )
            store.add(obs)

        return store

    def _load_trees(self, user_ref) -> TreeStore:
        """Load trees from subcollection."""
        store = TreeStore()
        tree_docs = user_ref.collection("trees").stream()

        for doc in tree_docs:
            data = doc.to_dict()
            from ..models.tree import Node

            tree = Tree(
                id=doc.id,
                root_value=data.get("root_value", ""),
                description=data.get("description", ""),
                root_node=Node.from_dict(data["root_node"]) if data.get("root_node") else None,
            )
            store.add(tree)

        return store

    # =========================================================================
    # Incremental Operations
    # =========================================================================

    def add_observation(self, user_id: str, obs: Observation) -> None:
        """Add a single observation."""
        self.db.collection("users").document(user_id)\
            .collection("observations").document(obs.id).set({
                "content": obs.content,
                "source_id": obs.source_id,
                "timestamp": obs.timestamp,
                "embedding": obs.embedding,
                "metadata": obs.metadata,
            })

    def add_observations(self, user_id: str, observations: list[Observation]) -> None:
        """Add multiple observations in batch."""
        user_ref = self.db.collection("users").document(user_id)
        obs_ref = user_ref.collection("observations")

        for i in range(0, len(observations), 500):
            batch = self.db.batch()
            for obs in observations[i:i+500]:
                batch.set(obs_ref.document(obs.id), {
                    "content": obs.content,
                    "source_id": obs.source_id,
                    "timestamp": obs.timestamp,
                    "embedding": obs.embedding,
                    "metadata": obs.metadata,
                })
            batch.commit()

    def save_tree(self, user_id: str, tree: Tree) -> None:
        """Save or update a single tree."""
        self.db.collection("users").document(user_id)\
            .collection("trees").document(tree.id).set({
                "root_value": tree.root_value,
                "description": tree.description,
                "root_node": tree.root_node.to_dict() if tree.root_node else None,
                "score": tree.score,
                "depth": tree.depth(),
            })

    def update_agents(self, user_id: str, agents: AgentSet) -> None:
        """Update agent allocations."""
        self.db.collection("users").document(user_id)\
            .collection("world_model").document("profile").update({
                "agents": agents.to_dict(),
                "updated_at": datetime.now(),
            })

    def get_tree(self, user_id: str, tree_id: str) -> Optional[Tree]:
        """Load a single tree."""
        doc = self.db.collection("users").document(user_id)\
            .collection("trees").document(tree_id).get()

        if not doc.exists:
            return None

        data = doc.to_dict()
        from ..models.tree import Node

        return Tree(
            id=doc.id,
            root_value=data.get("root_value", ""),
            description=data.get("description", ""),
            root_node=Node.from_dict(data["root_node"]) if data.get("root_node") else None,
        )

    def list_trees(self, user_id: str) -> list[dict]:
        """List all trees (metadata only, not full structure)."""
        docs = self.db.collection("users").document(user_id)\
            .collection("trees").stream()

        return [
            {
                "id": doc.id,
                "root_value": doc.get("root_value"),
                "score": doc.get("score"),
                "depth": doc.get("depth"),
            }
            for doc in docs
        ]

    def delete_world_model(self, user_id: str) -> None:
        """Delete entire world model for a user."""
        user_ref = self.db.collection("users").document(user_id)

        # Delete subcollections
        self._delete_collection(user_ref.collection("observations"))
        self._delete_collection(user_ref.collection("trees"))
        self._delete_collection(user_ref.collection("world_model"))

    def _delete_collection(self, coll_ref, batch_size: int = 500) -> None:
        """Delete all documents in a collection."""
        docs = coll_ref.limit(batch_size).stream()
        deleted = 0

        for doc in docs:
            doc.reference.delete()
            deleted += 1

        if deleted >= batch_size:
            self._delete_collection(coll_ref, batch_size)
