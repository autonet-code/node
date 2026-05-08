"""
Extracts atomic observations from documents using Claude CLI.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..models.observation import Observation, ObservationStore
from ..models.evidence import Source


EXTRACTION_PROMPT = """You are extracting atomic observations about a person from a document.

## What is an Observation?

An observation is a single, atomic fact about the person. Not an interpretation, not a judgment - just a fact that could be true or false.

Rules:
1. ONE fact per observation
2. Maximum ~280 characters (like a tweet)
3. Third person ("He...", "Andrei...")
4. No value judgments - just what IS
5. Specific, not vague

## Examples

Good observations:
- "He spent 10 years building governance frameworks"
- "He lives paycheck to paycheck at age 42"
- "He built the Jurisdiction system with fractal DAO topology"
- "ViraTrace was rejected by the EU for not being profitable"
- "He uses ayahuasca for psychological calibration, 50+ sessions over 10 years"
- "His closest relationship is with Raluca, college sweethearts who reconnected"

Bad observations:
- "He is dedicated" (too vague, interpretation)
- "He spent 10 years building governance frameworks because he believes in decentralization and wants to change the world" (multiple facts, too long)
- "I think he distrusts institutions" (interpretation, first person)

## Output Format

Return a JSON array of observation strings:

```json
[
  "Observation 1 text here",
  "Observation 2 text here",
  ...
]
```

Extract as many observations as you can find. Be thorough. Every distinct fact is worth capturing.

## Document

{document}

---

Extract observations as JSON array. Return ONLY the JSON, no other text."""


class ObservationExtractor:
    """Extracts atomic observations from documents using Claude CLI."""

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, document: str, source: Source) -> list[Observation]:
        """
        Extract observations from a document.

        Args:
            document: Text content to analyze
            source: Source metadata

        Returns:
            List of Observation objects
        """
        prompt = EXTRACTION_PROMPT.format(document=document)

        # Write prompt to temp file
        prompt_file = self.work_dir / f"obs_prompt_{source.id[:8]}.txt"
        output_file = self.work_dir / f"obs_output_{source.id[:8]}.txt"

        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            # Call Claude CLI
            if os.name == 'nt':  # Windows
                cmd = f'type "{prompt_file}" | claude -p --dangerously-skip-permissions'
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
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
            output_file.write_text(response_text, encoding="utf-8")

        finally:
            try:
                prompt_file.unlink()
            except:
                pass

        # Parse JSON
        observations_text = self._parse_json_response(response_text)

        # Convert to Observation objects
        observations = []
        for text in observations_text:
            obs = Observation(
                content=text,
                source_id=source.id,
                metadata={"source_name": source.name}
            )
            observations.append(obs)

        return observations

    def _parse_json_response(self, text: str) -> list[str]:
        """Extract JSON array of strings from response"""
        start = text.find('[')
        end = text.rfind(']') + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON array found in response: {text[:500]}...")

        json_str = text[start:end]
        return json.loads(json_str)

    def extract_to_store(
        self,
        document: str,
        source: Source,
        store: ObservationStore
    ) -> tuple[int, int]:
        """
        Extract observations and add to store.

        Returns:
            (new_count, duplicate_count)
        """
        observations = self.extract(document, source)

        new_count = 0
        dup_count = 0

        for obs in observations:
            _, is_new = store.add(obs)
            if is_new:
                new_count += 1
            else:
                dup_count += 1

        return new_count, dup_count
