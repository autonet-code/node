from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


@dataclass
class Source:
    """A source document (conversation, document, etc.)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: Optional[str] = None
    content_hash: Optional[str] = None
    ingested_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidencePointer:
    """Points to specific evidence within a source that supports a deviation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    excerpt: str = ""  # The relevant quote/passage
    context: str = ""  # What was happening when this was observed
    timestamp: Optional[datetime] = None  # When in the source timeline
    confidence: float = 1.0  # How strongly this evidence supports the deviation

    def __repr__(self):
        excerpt_preview = self.excerpt[:50] + "..." if len(self.excerpt) > 50 else self.excerpt
        return f"Evidence({excerpt_preview})"
