"""
Phase 1: Cross-Node Knowledge Transfer Evaluation

Tests whether federated JEPA training transfers knowledge between nodes,
or only enforces alignment.

Setup:
  - 3 simulated nodes, each with domain-specific text data
  - Node A: medical text, Node B: legal text, Node C: technical text
  - Each trains a TextJEPA locally for N steps
  - Weight deltas merged via FedAvg
  - After merge, test whether each node can embed out-of-domain text
    closer to the correct domain cluster

Success criteria:
  - Cross-domain retrieval@5 improves after FedAvg merge
  - Embedding variance stays high (no collapse)

Uses TextJEPA (lightweight, no LLM download needed). If knowledge
transfers at this level, it will transfer with the LLM backbone too —
same architecture, richer input features.
"""

import copy
import logging
import sys
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# Add project root
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from nodes.common.text_jepa import TextJEPAConfig, TextJEPATrainer
from nodes.common.text_data import TextMasker
from nodes.common.ml import aggregate_weight_deltas
from nodes.common.tokenizer import SimpleTokenizer, PAD_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific synthetic corpora
# ---------------------------------------------------------------------------

MEDICAL_TEXTS = [
    "The patient presented with acute myocardial infarction and elevated troponin levels requiring immediate percutaneous coronary intervention.",
    "Differential diagnosis includes pneumonia, pulmonary embolism, and congestive heart failure based on chest radiograph findings.",
    "Administer intravenous heparin bolus followed by continuous infusion and monitor activated partial thromboplastin time every six hours.",
    "The tumor staging revealed T3N1M0 adenocarcinoma with positive surgical margins necessitating adjuvant chemotherapy with FOLFOX regimen.",
    "Neurological examination showed decreased deep tendon reflexes and positive Babinski sign consistent with upper motor neuron lesion.",
    "Echocardiography demonstrates left ventricular ejection fraction of thirty-five percent with moderate mitral regurgitation.",
    "Blood cultures grew methicillin-resistant Staphylococcus aureus sensitive to vancomycin and linezolid for treatment.",
    "The pathology report confirmed high-grade dysplasia in the colonic polyp requiring surveillance colonoscopy in six months.",
    "Spirometry results show FEV1 of sixty percent predicted consistent with moderate chronic obstructive pulmonary disease.",
    "Magnetic resonance imaging revealed a two centimeter enhancing lesion in the right frontal lobe with surrounding vasogenic edema.",
    "Serum creatinine elevated to three point five indicating acute kidney injury likely secondary to dehydration and nephrotoxic medications.",
    "Glucose tolerance test showed fasting glucose of one hundred forty milligrams per deciliter consistent with impaired fasting glucose.",
    "Lumbar puncture showed elevated opening pressure with lymphocytic pleocytosis suggesting viral meningitis.",
    "The dermatological biopsy confirmed basal cell carcinoma with clear margins no further excision required.",
    "Hemoglobin A1C of nine point two percent indicates poorly controlled type two diabetes requiring insulin initiation.",
]

LEGAL_TEXTS = [
    "The defendant filed a motion for summary judgment arguing that no genuine issue of material fact exists under Federal Rule of Civil Procedure 56.",
    "Pursuant to the Fourth Amendment, the warrantless search of the vehicle was found unconstitutional and all evidence obtained thereby is inadmissible.",
    "The arbitration clause in the employment contract is enforceable under the Federal Arbitration Act and compels binding arbitration of all disputes.",
    "Under the doctrine of respondeat superior, the employer is vicariously liable for the tortious acts of its employee committed within the scope of employment.",
    "The statute of limitations for breach of contract claims in this jurisdiction is six years from the date of the alleged breach.",
    "The court granted the preliminary injunction finding that plaintiff demonstrated likelihood of success on the merits and irreparable harm.",
    "Intellectual property rights in the software are assigned to the corporation pursuant to the work-for-hire doctrine under the Copyright Act.",
    "The non-compete agreement is unenforceable as it lacks reasonable geographic and temporal limitations required under state law.",
    "Breach of fiduciary duty claim requires showing that defendant owed plaintiff a duty of loyalty and care which was violated.",
    "The merger agreement contains a material adverse change clause that permits termination if the target company suffers significant deterioration.",
    "Due process requires that the defendant receive adequate notice and a meaningful opportunity to be heard before deprivation of property.",
    "The class action certification was denied because individual issues of fact predominate over common questions of law.",
    "Under the Uniform Commercial Code section two-six-oh-one, buyer may reject goods that fail to conform to the contract.",
    "The appellate court reversed the lower court holding that the trial judge abused discretion in excluding expert testimony.",
    "Shareholder derivative suits require demand futility showing that the board of directors cannot impartially consider the demand.",
]

TECHNICAL_TEXTS = [
    "The distributed consensus protocol uses a Byzantine fault-tolerant algorithm that tolerates up to one-third malicious nodes in the network.",
    "Memory allocation in the garbage collector follows a generational strategy with young and old generation spaces to minimize pause times.",
    "The neural network architecture employs multi-head attention with eight heads and a hidden dimension of five hundred twelve.",
    "Database query optimization uses cost-based analysis with histogram statistics on column cardinality to select optimal join orderings.",
    "The load balancer implements consistent hashing with virtual nodes to distribute traffic evenly across the backend server cluster.",
    "Cryptographic signatures use elliptic curve digital signature algorithm with the secp256k1 curve for transaction authentication.",
    "The container orchestration system scales pods horizontally based on CPU utilization metrics collected every fifteen seconds.",
    "Compiler optimization passes include dead code elimination, loop unrolling, and register allocation using graph coloring.",
    "The message queue guarantees exactly-once delivery semantics through idempotent consumers and deduplication at the broker level.",
    "WebSocket connections maintain persistent bidirectional communication channels between client and server with heartbeat ping-pong frames.",
    "The B-tree index structure maintains balanced height with a branching factor of two hundred fifty-six for efficient disk page access patterns.",
    "Sharding the database across sixteen partitions uses consistent hash ring with replication factor three for fault tolerance.",
    "The CI/CD pipeline runs unit tests, integration tests, and security scanning before deploying containerized services to the staging environment.",
    "Thread pool executor manages a bounded queue of tasks with configurable core and maximum pool sizes to prevent resource exhaustion.",
    "The reverse proxy terminates TLS connections and forwards decrypted traffic to upstream services on the internal network.",
]

# Held-out queries for evaluation (5 per domain)
MEDICAL_QUERIES = [
    "What are the symptoms of acute myocardial infarction and how is it diagnosed?",
    "Describe the treatment protocol for methicillin-resistant Staphylococcus aureus bacteremia.",
    "What does an ejection fraction of thirty percent indicate on echocardiography?",
    "How is chronic obstructive pulmonary disease classified based on spirometry results?",
    "What is the significance of elevated troponin levels in emergency department patients?",
]

LEGAL_QUERIES = [
    "What are the requirements for filing a motion for summary judgment in federal court?",
    "Under what circumstances can a non-compete agreement be deemed unenforceable?",
    "What constitutes a breach of fiduciary duty and what elements must be proven?",
    "How does the doctrine of respondeat superior apply to employer liability?",
    "What is the standard for granting a preliminary injunction in civil litigation?",
]

TECHNICAL_QUERIES = [
    "How does a Byzantine fault-tolerant consensus protocol handle malicious nodes?",
    "What strategies does a generational garbage collector use to minimize pause times?",
    "How does consistent hashing distribute load across servers in a cluster?",
    "What optimizations does a cost-based query optimizer use for join ordering?",
    "How do container orchestration systems handle horizontal scaling decisions?",
]

DOMAIN_NAMES = ["medical", "legal", "technical"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize_texts(texts: List[str], max_len: int = 512) -> Dict[str, torch.Tensor]:
    """Byte-level tokenize a list of texts into padded tensors."""
    tokenizer = SimpleTokenizer()
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)[:max_len]
        all_ids.append(ids)

    # Pad to max length in batch
    max_actual = max(len(ids) for ids in all_ids)
    padded = []
    masks = []
    for ids in all_ids:
        pad_len = max_actual - len(ids)
        padded.append(ids + [PAD_ID] * pad_len)
        masks.append([True] * len(ids) + [False] * pad_len)

    return {
        "token_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
    }


def mean_pool_embeddings(
    trainer: TextJEPATrainer,
    texts: List[str],
    max_len: int = 512,
) -> torch.Tensor:
    """Encode texts and mean-pool to get (N, embed_dim) embeddings."""
    batch = tokenize_texts(texts, max_len)
    trainer.model.eval()
    with torch.no_grad():
        token_ids = batch["token_ids"].to(trainer.device)
        # Use context encoder (full sequence, no masking)
        embeddings = trainer.model.context_encoder(token_ids)  # (N, S, D)
        # Mean pool over sequence
        mask = batch["attention_mask"].to(trainer.device).unsqueeze(-1).float()
        # Embeddings might be shorter if masking was applied; use full seq
        seq_len = embeddings.shape[1]
        mask = mask[:, :seq_len, :]
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled  # (N, embed_dim)


def retrieval_at_k(
    query_embeds: torch.Tensor,
    corpus_embeds: torch.Tensor,
    query_labels: List[int],
    corpus_labels: List[int],
    k: int = 5,
) -> float:
    """Compute retrieval@k: fraction of queries whose top-k neighbors
    contain at least one same-domain corpus item.

    Args:
        query_embeds: (Q, D) query embeddings
        corpus_embeds: (C, D) corpus embeddings
        query_labels: domain label per query (0, 1, 2)
        corpus_labels: domain label per corpus item
        k: number of neighbors to check

    Returns:
        retrieval@k score in [0, 1]
    """
    # Cosine similarity
    query_norm = F.normalize(query_embeds, dim=1)
    corpus_norm = F.normalize(corpus_embeds, dim=1)
    sim = query_norm @ corpus_norm.T  # (Q, C)

    hits = 0
    for i in range(len(query_labels)):
        topk_idx = sim[i].topk(min(k, sim.shape[1])).indices
        topk_labels = [corpus_labels[j] for j in topk_idx]
        if query_labels[i] in topk_labels:
            hits += 1

    return hits / len(query_labels)


def embedding_variance(embeds: torch.Tensor) -> float:
    """Mean pairwise cosine distance — high = diverse, low = collapsed."""
    normed = F.normalize(embeds, dim=1)
    sim_matrix = normed @ normed.T
    # Exclude diagonal
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    mean_sim = sim_matrix[mask].mean().item()
    return 1.0 - mean_sim  # Convert similarity to distance


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    training_steps: int = 80,
    learning_rate: float = 3e-4,
    embed_dim: int = 128,
    encoder_depth: int = 3,
    max_seq_length: int = 256,
    k: int = 5,
):
    """Run the full Phase 1 cross-node knowledge transfer evaluation."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Configuration ──
    config = TextJEPAConfig(
        embed_dim=embed_dim,
        num_heads=4,
        encoder_depth=encoder_depth,
        predictor_depth=2,
        predictor_embed_dim=embed_dim // 2,
        max_seq_length=max_seq_length,
        ema_momentum=0.996,
    )

    # ── Create 3 "nodes" with identical initial weights ──
    log.info("Initializing 3 nodes with identical weights...")
    seed_trainer = TextJEPATrainer(config, device=device)
    initial_weights = seed_trainer.get_weights()

    trainers = []
    for i in range(3):
        t = TextJEPATrainer(config, device=device)
        t.load_weights(copy.deepcopy(initial_weights))
        trainers.append(t)

    # Domain data: Node 0 = medical, Node 1 = legal, Node 2 = technical
    domain_texts = [MEDICAL_TEXTS, LEGAL_TEXTS, TECHNICAL_TEXTS]
    domain_queries = [MEDICAL_QUERIES, LEGAL_QUERIES, TECHNICAL_QUERIES]

    # ── Pre-training baseline: measure cross-domain retrieval ──
    log.info("\n=== PRE-TRAINING BASELINE ===")
    baseline_scores = measure_cross_domain_retrieval(
        trainers, domain_texts, domain_queries, k=k,
    )

    # ── Train each node on its domain ──
    log.info("\n=== TRAINING (per-node, domain-specific) ===")
    for node_idx, (trainer, texts) in enumerate(zip(trainers, domain_texts)):
        log.info("Training Node %d (%s) for %d steps...",
                 node_idx, DOMAIN_NAMES[node_idx], training_steps)
        optimizer = torch.optim.AdamW(
            [p for p in trainer.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        batch = tokenize_texts(texts, max_seq_length)

        for step in range(training_steps):
            metrics = trainer.train_step(batch, optimizer)
            if (step + 1) % 20 == 0:
                log.info("  Node %d step %d: loss=%.4f cos_sim=%.4f",
                         node_idx, step + 1, metrics["loss"], metrics["cosine_similarity"])

    # ── Post-training, pre-merge: measure cross-domain retrieval ──
    log.info("\n=== POST-TRAINING, PRE-MERGE ===")
    pre_merge_scores = measure_cross_domain_retrieval(
        trainers, domain_texts, domain_queries, k=k,
    )

    # ── FedAvg merge ──
    log.info("\n=== FEDAVG MERGE ===")

    # Compute weight deltas relative to initial weights
    deltas = []
    for trainer in trainers:
        current = trainer.get_weights()
        delta = {}
        for key in initial_weights:
            delta[key] = (current[key] - initial_weights[key]).tolist()
        deltas.append(delta)

    # Merge with equal weights (each node contributed equally)
    merged_delta = aggregate_weight_deltas(deltas, weights=[1.0, 1.0, 1.0])

    # Apply merged delta to initial weights → new global model
    merged_weights = {}
    for key in initial_weights:
        merged_weights[key] = initial_weights[key] + merged_delta[key]

    # Load merged weights into all nodes
    for trainer in trainers:
        trainer.load_weights(copy.deepcopy(merged_weights))

    # ── Post-merge: measure cross-domain retrieval ──
    log.info("\n=== POST-MERGE ===")
    post_merge_scores = measure_cross_domain_retrieval(
        trainers, domain_texts, domain_queries, k=k,
    )

    # ── Embedding collapse check ──
    log.info("\n=== EMBEDDING COLLAPSE CHECK ===")
    all_texts = MEDICAL_TEXTS + LEGAL_TEXTS + TECHNICAL_TEXTS
    for node_idx, trainer in enumerate(trainers):
        embeds = mean_pool_embeddings(trainer, all_texts, max_seq_length)
        var = embedding_variance(embeds)
        log.info("Node %d (%s) embedding variance: %.4f %s",
                 node_idx, DOMAIN_NAMES[node_idx], var,
                 "(HEALTHY)" if var > 0.1 else "(COLLAPSED)")

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("SUMMARY: Cross-Domain Retrieval@%d", k)
    log.info("=" * 60)
    log.info("%-25s %10s %10s %10s", "", "Baseline", "Pre-Merge", "Post-Merge")
    log.info("-" * 60)

    for node_idx in range(3):
        domain = DOMAIN_NAMES[node_idx]
        b = baseline_scores[node_idx]
        pre = pre_merge_scores[node_idx]
        post = post_merge_scores[node_idx]
        delta = post - b
        log.info("Node %d (%-10s) %8.1f%% %9.1f%% %9.1f%%  (%+.1f%%)",
                 node_idx, domain, b * 100, pre * 100, post * 100, delta * 100)

    avg_baseline = sum(baseline_scores) / 3
    avg_pre = sum(pre_merge_scores) / 3
    avg_post = sum(post_merge_scores) / 3
    avg_delta = avg_post - avg_baseline

    log.info("-" * 60)
    log.info("%-25s %8.1f%% %9.1f%% %9.1f%%  (%+.1f%%)",
             "Average", avg_baseline * 100, avg_pre * 100, avg_post * 100, avg_delta * 100)
    log.info("=" * 60)

    if avg_delta > 0.10:
        log.info("RESULT: Knowledge transfer DETECTED (>10%% improvement)")
    elif avg_delta > 0.0:
        log.info("RESULT: Marginal improvement (%.1f%%) — needs more training or architectural changes", avg_delta * 100)
    else:
        log.info("RESULT: No knowledge transfer detected — JEPA may only be learning alignment, not capability")

    return {
        "baseline": baseline_scores,
        "pre_merge": pre_merge_scores,
        "post_merge": post_merge_scores,
        "improvement": avg_delta,
    }


def measure_cross_domain_retrieval(
    trainers: List[TextJEPATrainer],
    domain_texts: List[List[str]],
    domain_queries: List[List[str]],
    k: int = 5,
) -> List[float]:
    """Measure how well each node retrieves out-of-domain content.

    For each node, we ask: can this node's encoder embed queries from
    OTHER domains and correctly match them to corpus items from those domains?

    Returns list of retrieval@k scores, one per node.
    """
    max_len = 256
    scores = []

    for node_idx, trainer in enumerate(trainers):
        # Build corpus from ALL domains (labeled)
        all_corpus_texts = []
        corpus_labels = []
        for d, texts in enumerate(domain_texts):
            all_corpus_texts.extend(texts)
            corpus_labels.extend([d] * len(texts))

        # Build queries from OTHER domains only (cross-domain test)
        cross_queries = []
        query_labels = []
        for d, queries in enumerate(domain_queries):
            if d != node_idx:  # Skip this node's own domain
                cross_queries.extend(queries)
                query_labels.extend([d] * len(queries))

        # Embed everything through this node's encoder
        corpus_embeds = mean_pool_embeddings(trainer, all_corpus_texts, max_len)
        query_embeds = mean_pool_embeddings(trainer, cross_queries, max_len)

        score = retrieval_at_k(query_embeds, corpus_embeds, query_labels, corpus_labels, k=k)
        log.info("  Node %d (%s) cross-domain retrieval@%d: %.1f%%",
                 node_idx, DOMAIN_NAMES[node_idx], k, score * 100)
        scores.append(score)

    return scores


if __name__ == "__main__":
    results = run_evaluation(
        training_steps=80,
        learning_rate=3e-4,
        embed_dim=128,
        encoder_depth=3,
        max_seq_length=256,
        k=5,
    )
