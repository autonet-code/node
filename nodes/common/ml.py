"""
Machine Learning Module for Autonet

Implements real PyTorch training for federated learning with:
- SimpleNet: Small CNN for MNIST (2 conv + 2 fc layers)
- CUDA acceleration when available
- Weight delta computation for FedAvg aggregation
- IPFS integration for model storage
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from typing import Dict, Any, Optional, Tuple
import io
import json
import time

logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    """
    Small CNN for MNIST classification.

    Architecture:
    - Conv2d(1, 16, 3x3) + ReLU + MaxPool2d(2x2)
    - Conv2d(16, 32, 3x3) + ReLU + MaxPool2d(2x2)
    - Flatten
    - Linear(32*5*5, 64) + ReLU
    - Linear(64, 10)

    Total parameters: ~11K
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Fully connected layers
        # After 2 MaxPool2d(2x2): 28x28 -> 14x14 -> 7x7
        # But with padding, it's 28x28 -> 14x14 -> 7x7
        # However, let's calculate: 28->14->7, but conv keeps same size
        # Actually: (28+2*1-3+1)/1+1 = 28, pool -> 14, conv -> 14, pool -> 7
        # Wait: conv(28, pad=1, k=3) = 28, pool(28, k=2) = 14
        #       conv(14, pad=1, k=3) = 14, pool(14, k=2) = 7
        # So: 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv block 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 32 * 7 * 7)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_device() -> torch.device:
    """Get the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mnist_data(
    train: bool = True,
    data_dir: str = "./data",
    batch_size: int = 32,
    num_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Load MNIST dataset with standard transformations.

    Args:
        train: Whether to load training or test data
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoader
        num_samples: Limit to first N samples (for fast demo training)

    Returns:
        DataLoader for MNIST data
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    dataset = datasets.MNIST(
        data_dir,
        train=train,
        download=True,
        transform=transform
    )

    # Limit dataset size for faster training
    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,  # Windows compatibility
    )

    return dataloader


def train_on_task(
    task_spec: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    global_model_cid: Optional[str] = None,
    ipfs_client = None,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    num_samples: int = 500,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train a model on a task and return weight delta for FedAvg.

    This performs real federated learning:
    1. Load global model (or initialize fresh)
    2. Train on local data subset
    3. Compute weight delta (new_weights - old_weights)
    4. Return delta for aggregation

    Args:
        task_spec: Task specification (contains training config)
        ground_truth: Ground truth data (optional, for verification)
        global_model_cid: CID of global model weights (None = fresh model)
        ipfs_client: IPFSClient for loading/saving weights
        epochs: Number of training epochs (keep low for speed)
        batch_size: Training batch size
        learning_rate: Learning rate for SGD
        num_samples: Number of samples to train on (subset of MNIST)

    Returns:
        (weight_delta_dict, metrics_dict)
    """
    device = get_device()
    start_time = time.time()

    # Initialize model
    model = SimpleNet().to(device)
    num_params = count_parameters(model)
    logger.info(f"Model initialized with {num_params:,} parameters")

    # Load global model weights if available
    initial_state_dict = None
    if global_model_cid and ipfs_client:
        try:
            initial_state_dict = load_weights(global_model_cid, ipfs_client)
            if initial_state_dict:
                model.load_state_dict(initial_state_dict)
                logger.info(f"Loaded global model from {global_model_cid}")
        except Exception as e:
            logger.warning(f"Could not load global model: {e}")

    # Save initial weights for delta computation
    if initial_state_dict is None:
        initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Get training data (subset of MNIST for fast training)
    try:
        train_loader = get_mnist_data(
            train=True,
            batch_size=batch_size,
            num_samples=num_samples
        )
        logger.info(f"Loaded {len(train_loader.dataset)} training samples")
    except Exception as e:
        logger.error(f"Failed to load MNIST data: {e}")
        # Return mock result on failure
        return _mock_training_result(task_spec)

    # Train model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_total += target.size(0)

        epoch_accuracy = epoch_correct / epoch_total
        epoch_avg_loss = epoch_loss / len(train_loader)
        total_loss += epoch_avg_loss
        correct += epoch_correct
        total += epoch_total

        logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_avg_loss:.4f}, Accuracy={epoch_accuracy:.4f}")

    training_time = time.time() - start_time

    # Compute weight delta (for FedAvg)
    final_state_dict = model.state_dict()
    weight_delta = {}
    for key in final_state_dict:
        weight_delta[key] = (final_state_dict[key] - initial_state_dict[key]).cpu().numpy().tolist()

    # Compute metrics
    metrics = {
        "loss": total_loss / epochs,
        "accuracy": correct / total,
        "training_time": training_time,
        "epochs": epochs,
        "num_samples": len(train_loader.dataset),
        "num_parameters": num_params,
        "device": str(device),
    }

    logger.info(f"Training completed in {training_time:.2f}s: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")

    return weight_delta, metrics


def _mock_training_result(task_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fallback mock training result if real training fails."""
    task_id = task_spec.get("task_id", 0)
    return {
        "task_id": task_id,
        "model_weights": f"mock_weights_task_{task_id}",
    }, {
        "loss": 0.15,
        "accuracy": 0.92,
        "training_time": 0.1,
        "epochs": 1,
        "num_samples": 500,
        "mock": True,
    }


def save_weights(state_dict: Dict[str, torch.Tensor], ipfs_client) -> Optional[str]:
    """
    Save model weights to IPFS.

    Args:
        state_dict: PyTorch state_dict (model weights)
        ipfs_client: IPFSClient instance

    Returns:
        CID of saved weights, or None on failure
    """
    try:
        # Convert tensors to lists for JSON serialization
        serializable_dict = {}
        for key, tensor in state_dict.items():
            serializable_dict[key] = tensor.cpu().numpy().tolist()

        # Save to IPFS
        cid = ipfs_client.add_json({
            "weights": serializable_dict,
            "format": "pytorch_state_dict",
        })

        logger.info(f"Saved weights to IPFS: {cid}")
        return cid

    except Exception as e:
        logger.error(f"Failed to save weights to IPFS: {e}")
        return None


def load_weights(cid: str, ipfs_client) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load model weights from IPFS.

    Args:
        cid: IPFS CID of weights
        ipfs_client: IPFSClient instance

    Returns:
        PyTorch state_dict, or None on failure
    """
    try:
        # Load from IPFS
        data = ipfs_client.get_json(cid)
        if not data:
            logger.error(f"Failed to load weights from IPFS: {cid}")
            return None

        # Convert lists back to tensors
        state_dict = {}
        weights = data.get("weights", {})
        for key, value in weights.items():
            state_dict[key] = torch.tensor(value)

        logger.info(f"Loaded weights from IPFS: {cid}")
        return state_dict

    except Exception as e:
        logger.error(f"Failed to load weights from IPFS: {e}")
        return None


def aggregate_weight_deltas(
    deltas: list,
    weights: Optional[list] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate weight deltas using FedAvg algorithm.

    FedAvg: weighted_avg = sum(weight_i * delta_i) / sum(weights)

    Args:
        deltas: List of weight delta dicts (from train_on_task)
        weights: Optional list of weights for each delta (e.g., num_samples)
                 If None, simple average is used

    Returns:
        Aggregated weight delta dict
    """
    if not deltas:
        return {}

    # Use uniform weights if not provided
    if weights is None:
        weights = [1.0] * len(deltas)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Aggregate deltas
    aggregated = {}
    for key in deltas[0].keys():
        # Convert to tensors and aggregate
        tensors = [torch.tensor(delta[key]) * w for delta, w in zip(deltas, normalized_weights)]
        aggregated[key] = sum(tensors)

    logger.info(f"Aggregated {len(deltas)} weight deltas with weights {weights}")
    return aggregated


def apply_weight_delta(
    base_state_dict: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Apply weight delta to base model weights.

    Args:
        base_state_dict: Base model state_dict
        delta: Weight delta to apply

    Returns:
        Updated state_dict
    """
    updated = {}
    for key in base_state_dict:
        if key in delta:
            updated[key] = base_state_dict[key] + delta[key]
        else:
            updated[key] = base_state_dict[key]

    return updated


def test_model(
    model_cid: str,
    ipfs_client,
    num_samples: int = 1000,
) -> Dict[str, float]:
    """
    Test a model on MNIST test set.

    Args:
        model_cid: CID of model weights
        ipfs_client: IPFSClient instance
        num_samples: Number of test samples

    Returns:
        Dict with test metrics
    """
    device = get_device()

    # Load model
    model = SimpleNet().to(device)
    state_dict = load_weights(model_cid, ipfs_client)
    if state_dict:
        model.load_state_dict(state_dict)
    else:
        logger.warning("Could not load model weights, testing with random weights")

    # Load test data
    test_loader = get_mnist_data(
        train=False,
        batch_size=32,
        num_samples=num_samples
    )

    # Test model
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = correct / total

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "num_samples": total,
    }

    logger.info(f"Test results: Loss={test_loss:.4f}, Accuracy={accuracy:.4f}")
    return metrics


if __name__ == "__main__":
    """Quick test of ML module."""
    logging.basicConfig(level=logging.INFO)

    from ipfs import IPFSClient

    print("=== Testing ML Module ===")

    # Initialize IPFS client
    ipfs = IPFSClient()

    # Test model creation
    print("\n1. Creating SimpleNet...")
    model = SimpleNet()
    num_params = count_parameters(model)
    print(f"   Model has {num_params:,} parameters")

    # Test training
    print("\n2. Training on MNIST subset...")
    task_spec = {"task_id": 1, "epochs": 1}
    weight_delta, metrics = train_on_task(
        task_spec=task_spec,
        ipfs_client=ipfs,
        epochs=1,
        num_samples=100,  # Very small for quick test
    )
    print(f"   Training metrics: {metrics}")

    # Test saving/loading
    print("\n3. Testing IPFS save/load...")
    test_model = SimpleNet()
    state_dict = test_model.state_dict()
    cid = save_weights(state_dict, ipfs)
    print(f"   Saved to IPFS: {cid}")

    loaded = load_weights(cid, ipfs)
    if loaded:
        print(f"   Loaded successfully, {len(loaded)} keys")

    print("\n=== ML Module Test Complete ===")
