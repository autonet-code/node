"""
Embeddings & NLI - semantic similarity and stance detection.

Two capabilities:
1. Embeddings: Convert text to vectors for topical similarity
2. NLI: Detect entailment/contradiction/neutral between statements
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Lazy loading to avoid slow import on module load
_embedding_model = None
_embedding_model_name = "all-MiniLM-L6-v2"  # Fast, 384 dims, good quality

_nli_model = None
_nli_tokenizer = None
_nli_model_name = "microsoft/deberta-base-mnli"  # Good balance of speed/accuracy


def _get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{_embedding_model_name}'...")
        _embedding_model = SentenceTransformer(_embedding_model_name)
        print("Embedding model loaded.")
    return _embedding_model


def _get_nli_model():
    """Lazy load the NLI model."""
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print(f"Loading NLI model '{_nli_model_name}'...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(_nli_model_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(_nli_model_name)
        _nli_model.eval()  # Set to inference mode
        print("NLI model loaded.")
    return _nli_model, _nli_tokenizer


def embed(text: str) -> np.ndarray:
    """Convert text to embedding vector."""
    model = _get_embedding_model()
    return model.encode(text, convert_to_numpy=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Convert multiple texts to embeddings (faster than one at a time)."""
    model = _get_embedding_model()
    return model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 to 1 for normalized vecs)."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Returns value between 0 (unrelated) and 1 (identical meaning).
    """
    emb1 = embed(text1)
    emb2 = embed(text2)
    # Cosine similarity ranges -1 to 1, shift to 0 to 1
    sim = cosine_similarity(emb1, emb2)
    return (sim + 1) / 2  # Now 0 to 1


class EmbeddingCache:
    """
    Cache embeddings for repeated similarity comparisons.

    Useful when comparing one concept against many nodes.
    """

    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding, using cache if available."""
        if text not in self._cache:
            self._cache[text] = embed(text)
        return self._cache[text]

    def similarity(self, text1: str, text2: str) -> float:
        """Compute similarity using cached embeddings."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        sim = cosine_similarity(emb1, emb2)
        return (sim + 1) / 2

    def preload(self, texts: list[str]):
        """Preload embeddings for a batch of texts."""
        # Filter out already cached
        to_embed = [t for t in texts if t not in self._cache]
        if to_embed:
            embeddings = embed_batch(to_embed)
            for text, emb in zip(to_embed, embeddings):
                self._cache[text] = emb

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache for convenience
_global_cache = EmbeddingCache()


def cached_similarity(text1: str, text2: str) -> float:
    """Compute similarity using global cache."""
    return _global_cache.similarity(text1, text2)


def preload_cache(texts: list[str]):
    """Preload global cache with texts."""
    _global_cache.preload(texts)


def clear_cache():
    """Clear global cache."""
    _global_cache.clear()


# =============================================================================
# NLI (Natural Language Inference) - Stance Detection
# =============================================================================

@dataclass
class NLIResult:
    """Result of NLI inference between premise and hypothesis."""
    premise: str
    hypothesis: str
    entailment: float      # Hypothesis follows from premise
    contradiction: float   # Hypothesis contradicts premise
    neutral: float         # No clear relationship

    @property
    def stance(self) -> str:
        """Most likely stance."""
        scores = {
            "entailment": self.entailment,
            "contradiction": self.contradiction,
            "neutral": self.neutral,
        }
        return max(scores, key=scores.get)

    @property
    def is_contradiction(self) -> bool:
        return self.contradiction > self.entailment and self.contradiction > self.neutral

    @property
    def is_entailment(self) -> bool:
        return self.entailment > self.contradiction and self.entailment > self.neutral

    @property
    def support_score(self) -> float:
        """
        Combined score for novelty: positive = supports, negative = contradicts.
        Range roughly -1 to 1.
        """
        return self.entailment - self.contradiction


def nli_inference(premise: str, hypothesis: str) -> NLIResult:
    """
    Determine logical relationship between premise and hypothesis.

    Args:
        premise: The established statement (e.g., existing belief in tree)
        hypothesis: The new statement to evaluate (e.g., incoming concept)

    Returns:
        NLIResult with entailment/contradiction/neutral probabilities
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer = _get_nli_model()

    # Tokenize the premise-hypothesis pair
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0]

    # DeBERTa-MNLI label order: contradiction, neutral, entailment
    return NLIResult(
        premise=premise,
        hypothesis=hypothesis,
        contradiction=float(probs[0]),
        neutral=float(probs[1]),
        entailment=float(probs[2]),
    )


def batch_nli_inference(premise: str, hypotheses: list[str]) -> list[NLIResult]:
    """
    Evaluate multiple hypotheses against a single premise efficiently.
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer = _get_nli_model()

    # Tokenize all pairs
    inputs = tokenizer(
        [premise] * len(hypotheses),
        hypotheses,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    # Build results
    results = []
    for i, hyp in enumerate(hypotheses):
        results.append(NLIResult(
            premise=premise,
            hypothesis=hyp,
            contradiction=float(probs[i][0]),
            neutral=float(probs[i][1]),
            entailment=float(probs[i][2]),
        ))

    return results


@dataclass
class SemanticRelation:
    """Combined topical similarity + stance analysis."""
    text1: str
    text2: str
    topical_similarity: float   # 0-1, how related are the topics
    support_score: float        # -1 to 1, contradiction to entailment
    nli_result: NLIResult

    @property
    def novelty_fit_score(self) -> float:
        """
        Combined score for novelty computation.

        High topical similarity + support = fits well (low novelty)
        High topical similarity + contradiction = high novelty (challenges beliefs)
        Low topical similarity = doesn't fit (moderate novelty)
        """
        if self.topical_similarity < 0.4:
            # Not even on topic - moderate resistance
            return 0.3

        # On topic - stance matters
        # support_score: -1 (contradiction) to +1 (entailment)
        # We want: entailment -> high fit (0.8), contradiction -> low fit (0.1)
        fit = 0.45 + (self.support_score * 0.35)
        return max(0.1, min(0.9, fit))


def analyze_relation(text1: str, text2: str) -> SemanticRelation:
    """
    Full semantic analysis: topical similarity + stance.

    Use this for novelty computation - it tells you both
    whether concepts are related AND whether they agree.
    """
    # Get topical similarity via embeddings
    topical_sim = cached_similarity(text1, text2)

    # Get stance via NLI
    nli = nli_inference(text1, text2)

    return SemanticRelation(
        text1=text1,
        text2=text2,
        topical_similarity=topical_sim,
        support_score=nli.support_score,
        nli_result=nli,
    )


def relation_fit_score(text1: str, text2: str) -> float:
    """
    Convenience function returning just the fit score.

    Use as similarity_fn in NoveltyComputer.
    """
    relation = analyze_relation(text1, text2)
    return relation.novelty_fit_score
