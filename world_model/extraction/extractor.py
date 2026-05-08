import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..models import DeviationNode, DeviationType, EvidencePointer, Source


EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing human behavior, psychology, and worldviews. Your task is to extract "deviations" - ways a specific person differs from baseline human expectations.

## What is a Deviation?

A deviation is NOT a fact about someone. It's a way they DIFFER from what you'd expect of a typical person.

For each deviation you identify:
1. Name the dimension (emergent - don't use predefined categories, describe what you're actually seeing)
2. State the baseline assumption (what would a typical person think/do/feel in this domain?)
3. Describe how this person differs
4. Estimate magnitude (0.0-1.0, how far from baseline)
5. Note the evidence (specific quotes or observations from the source)

## Deviation Types

Categorize each deviation as one of:
- epistemic: How beliefs form, what sources are trusted, confidence calibration
- motivational: What drives action, goal priorities, time horizons, risk tolerance
- relational: How others are modeled, trust dynamics, attachment patterns
- attentional: What gets noticed, salience patterns, what triggers interest
- axiological: Values, what matters, meaning-making structures
- self_model: Identity, agency beliefs, narrative style
- behavioral: Action patterns, habits, coping mechanisms
- cognitive: Reasoning style, problem-solving approaches

## Output Format

Return a JSON array of deviations:
```json
[
  {
    "dimension": "string - emergent label for this deviation",
    "deviation_type": "epistemic|motivational|relational|attentional|axiological|self_model|behavioral|cognitive",
    "baseline_assumption": "string - what typical person would think/do",
    "deviation_description": "string - how this person differs",
    "magnitude": 0.0-1.0,
    "confidence": 0.0-1.0,
    "evidence": [
      {
        "excerpt": "string - direct quote or paraphrase from source",
        "context": "string - what was happening"
      }
    ],
    "tags": ["optional", "tags"],
    "parent_dimension": "string or null - if this is a sub-deviation of a broader pattern"
  }
]
```

## Guidelines

- Be specific. "Has unusual beliefs" is too vague. "Believes nation-states are obsolete infrastructure" is specific.
- The baseline should be genuinely typical, not strawman. Steel-man the average.
- Magnitude should reflect how rare this deviation is. Common variations = low magnitude. Rare positions = high magnitude.
- Include direct evidence. No deviation without supporting quotes/observations.
- Look for PATTERNS, not just isolated facts. A single statement isn't a deviation. Repeated themes are.
- Consider what's IMPLIED, not just stated. Behavior reveals worldview.
- Identify hierarchical relationships. Some deviations are sub-cases of broader deviations.
"""


EXTRACTION_USER_PROMPT = """Analyze the following document about a person and extract their deviations from baseline human expectations.

Focus on what makes this person DISTINCTIVE - not facts about them, but ways they differ from what you'd expect of a typical person.

Look for:
- Unusual beliefs or epistemic patterns
- Atypical goals or motivations
- Non-standard relationship patterns
- Different value weightings
- Distinctive reasoning or behavioral patterns

Document:
---
{document}
---

Extract the deviations as JSON. Be thorough but focused on genuine deviations, not every detail. Return ONLY the JSON array, no other text."""


class DeviationExtractor:
    """
    Extracts deviation nodes from source documents using Claude CLI.

    Uses the local Claude Code CLI installation, piping prompts through
    `claude -p --dangerously-skip-permissions` to leverage existing subscription.
    """

    def __init__(self, work_dir: Optional[str] = None):
        """
        Initialize extractor.

        Args:
            work_dir: Directory for temp files. Defaults to system temp.
        """
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, document: str, source: Source) -> list[DeviationNode]:
        """
        Extract deviations from a document using Claude CLI.

        Args:
            document: The text content to analyze
            source: Source metadata for evidence tracking

        Returns:
            List of DeviationNode objects
        """
        # Build the full prompt
        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

---

{EXTRACTION_USER_PROMPT.format(document=document)}"""

        # Write prompt to temp file
        prompt_file = self.work_dir / f"extract_prompt_{source.id[:8]}.txt"
        output_file = self.work_dir / f"extract_output_{source.id[:8]}.txt"

        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            # Call Claude CLI
            # Using shell=True on Windows with type command to pipe
            if os.name == 'nt':  # Windows
                cmd = f'type "{prompt_file}" | claude -p --dangerously-skip-permissions'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=str(self.work_dir)
                )
            else:  # Unix
                with open(prompt_file, 'r') as f:
                    result = subprocess.run(
                        ['claude', '-p', '--dangerously-skip-permissions'],
                        stdin=f,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=str(self.work_dir)
                    )

            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI failed: {result.stderr}")

            response_text = result.stdout

            # Save output for debugging
            output_file.write_text(response_text, encoding="utf-8")

        finally:
            # Clean up prompt file
            try:
                prompt_file.unlink()
            except:
                pass

        # Parse JSON from response
        deviations_data = self._parse_json_response(response_text)

        # Convert to DeviationNode objects
        nodes = []
        dimension_to_id = {}  # Track for parent relationships

        for dev_data in deviations_data:
            node = self._create_node(dev_data, source)
            nodes.append(node)
            dimension_to_id[dev_data.get("dimension", "")] = node.id

        # Wire up parent relationships
        for i, dev_data in enumerate(deviations_data):
            parent_dim = dev_data.get("parent_dimension")
            if parent_dim and parent_dim in dimension_to_id:
                nodes[i].parent_id = dimension_to_id[parent_dim]

        return nodes

    def _parse_json_response(self, text: str) -> list[dict]:
        """Extract JSON array from LLM response"""
        # Try to find JSON array in response
        start = text.find('[')
        end = text.rfind(']') + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found in response: {text[:500]}...")

        json_str = text[start:end]
        return json.loads(json_str)

    def _create_node(self, data: dict, source: Source) -> DeviationNode:
        """Convert parsed JSON to DeviationNode"""
        # Parse deviation type
        type_str = data.get("deviation_type", "epistemic")
        try:
            deviation_type = DeviationType(type_str)
        except ValueError:
            deviation_type = DeviationType.EPISTEMIC

        # Create evidence pointers
        evidence = []
        for ev in data.get("evidence", []):
            evidence.append(EvidencePointer(
                source_id=source.id,
                excerpt=ev.get("excerpt", ""),
                context=ev.get("context", "")
            ))

        return DeviationNode(
            dimension=data.get("dimension", ""),
            deviation_type=deviation_type,
            baseline_assumption=data.get("baseline_assumption", ""),
            deviation_description=data.get("deviation_description", ""),
            magnitude=float(data.get("magnitude", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            evidence=evidence,
            tags=data.get("tags", [])
        )

    def extract_incremental(
        self,
        document: str,
        source: Source,
        existing_nodes: list[DeviationNode]
    ) -> tuple[list[DeviationNode], list[DeviationNode]]:
        """
        Extract deviations, comparing against existing nodes.

        Returns:
            (new_nodes, reinforced_existing_nodes)
        """
        new_nodes = self.extract(document, source)

        # Try to match against existing
        reinforced = []
        truly_new = []

        existing_keys = {n.similarity_key(): n for n in existing_nodes}

        for node in new_nodes:
            key = node.similarity_key()
            if key in existing_keys:
                # Reinforce existing node
                existing = existing_keys[key]
                for ev in node.evidence:
                    existing.reinforce(ev)
                reinforced.append(existing)
            else:
                truly_new.append(node)

        return truly_new, reinforced


# Alternative: API-based extractor for when you have an API key
class APIDeviationExtractor(DeviationExtractor):
    """
    Extracts deviations using the Anthropic API directly.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        super().__init__()
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            self.model = model
            self._use_api = True
        except ImportError:
            raise ImportError("anthropic package required for API extractor: pip install anthropic")

    def extract(self, document: str, source: Source) -> list[DeviationNode]:
        """Extract deviations using API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_USER_PROMPT.format(document=document)
                }
            ]
        )

        response_text = response.content[0].text
        deviations_data = self._parse_json_response(response_text)

        nodes = []
        dimension_to_id = {}

        for dev_data in deviations_data:
            node = self._create_node(dev_data, source)
            nodes.append(node)
            dimension_to_id[dev_data.get("dimension", "")] = node.id

        for i, dev_data in enumerate(deviations_data):
            parent_dim = dev_data.get("parent_dimension")
            if parent_dim and parent_dim in dimension_to_id:
                nodes[i].parent_id = dimension_to_id[parent_dim]

        return nodes
