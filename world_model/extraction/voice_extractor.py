"""
Extracts voice/style patterns from text corpus.
Complements ObservationExtractor - captures HOW someone says things, not WHAT they say.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VoiceProfile:
    """
    Captures stylistic patterns of a person's communication.
    Used alongside World Model to generate authentic voice.
    """

    # Core voice characteristics
    tone_descriptors: list[str] = field(default_factory=list)  # e.g., ["sardonic", "philosophical", "irreverent"]
    register_range: str = ""  # e.g., "alternates high intellectual and crude humor"

    # Structural patterns
    sentence_patterns: list[str] = field(default_factory=list)  # e.g., ["aphoristic single-sentence", "setup-punchline"]
    opening_patterns: list[str] = field(default_factory=list)  # how they start statements
    closing_patterns: list[str] = field(default_factory=list)  # how they end statements

    # Vocabulary fingerprint
    characteristic_phrases: list[str] = field(default_factory=list)  # recurring expressions
    vocabulary_notes: list[str] = field(default_factory=list)  # e.g., "uses technical jargon freely"

    # Rhetorical devices
    rhetorical_devices: list[str] = field(default_factory=list)  # e.g., ["rhetorical questions", "self-deprecation"]
    humor_patterns: list[str] = field(default_factory=list)  # how they're funny

    # Engagement style
    confrontation_style: str = ""  # how they disagree/challenge
    agreement_style: str = ""  # how they affirm

    # Language mixing
    language_notes: str = ""  # e.g., "switches to Romanian for emotional/cultural content"

    # Sample exemplars - actual tweets that perfectly capture the voice
    exemplar_tweets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tone_descriptors": self.tone_descriptors,
            "register_range": self.register_range,
            "sentence_patterns": self.sentence_patterns,
            "opening_patterns": self.opening_patterns,
            "closing_patterns": self.closing_patterns,
            "characteristic_phrases": self.characteristic_phrases,
            "vocabulary_notes": self.vocabulary_notes,
            "rhetorical_devices": self.rhetorical_devices,
            "humor_patterns": self.humor_patterns,
            "confrontation_style": self.confrontation_style,
            "agreement_style": self.agreement_style,
            "language_notes": self.language_notes,
            "exemplar_tweets": self.exemplar_tweets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**data)

    def to_system_prompt_section(self) -> str:
        """Generate a system prompt section for voice styling."""
        lines = ["## Voice & Style Guide", ""]

        if self.tone_descriptors:
            lines.append(f"**Tone**: {', '.join(self.tone_descriptors)}")

        if self.register_range:
            lines.append(f"**Register**: {self.register_range}")

        if self.sentence_patterns:
            lines.append(f"**Sentence patterns**: {'; '.join(self.sentence_patterns)}")

        if self.rhetorical_devices:
            lines.append(f"**Rhetorical devices**: {', '.join(self.rhetorical_devices)}")

        if self.humor_patterns:
            lines.append(f"**Humor style**: {'; '.join(self.humor_patterns)}")

        if self.confrontation_style:
            lines.append(f"**When disagreeing**: {self.confrontation_style}")

        if self.characteristic_phrases:
            lines.append(f"**Characteristic expressions**: {', '.join(self.characteristic_phrases[:10])}")

        if self.language_notes:
            lines.append(f"**Language**: {self.language_notes}")

        if self.exemplar_tweets:
            lines.append("")
            lines.append("**Voice exemplars** (match this energy):")
            for tweet in self.exemplar_tweets[:10]:
                lines.append(f'- "{tweet}"')

        return "\n".join(lines)


VOICE_EXTRACTION_PROMPT = """You are analyzing a person's writing style to extract their unique voice patterns.

## Your Task

Analyze these tweets/posts and extract the distinctive voice characteristics. Focus on HOW they communicate, not WHAT they're saying.

## What to Extract

1. **Tone descriptors** (3-5 adjectives): What's the overall feel? (e.g., sardonic, earnest, irreverent, philosophical)

2. **Register range**: Do they stick to one register or mix high/low? (e.g., "mixes academic vocabulary with crude humor")

3. **Sentence patterns**: What structures do they favor?
   - Aphoristic single sentences?
   - Setup-then-subvert?
   - Questions followed by answers?
   - Stream of consciousness?

4. **Opening patterns**: How do they start statements? (e.g., "Often opens with a provocative claim")

5. **Closing patterns**: How do they end? (e.g., "Ends with self-deprecating aside")

6. **Characteristic phrases**: Recurring expressions or verbal tics

7. **Vocabulary notes**: Technical jargon? Slang? Formal? Neologisms?

8. **Rhetorical devices**: What techniques do they use?
   - Rhetorical questions
   - Irony/sarcasm
   - Hyperbole
   - Self-deprecation
   - Direct address

9. **Humor patterns**: How are they funny?
   - Absurdist
   - Dark
   - Observational
   - Self-deprecating
   - Wordplay

10. **Confrontation style**: How do they disagree or challenge others?

11. **Agreement style**: How do they affirm or support?

12. **Language notes**: Any multilingual patterns?

13. **Exemplar tweets**: Pick 10-15 tweets that PERFECTLY capture their voice - the ones you'd show someone to instantly understand how this person communicates.

## Output Format

Return a JSON object:

```json
{{
  "tone_descriptors": ["descriptor1", "descriptor2", ...],
  "register_range": "description of register usage",
  "sentence_patterns": ["pattern1", "pattern2", ...],
  "opening_patterns": ["pattern1", "pattern2", ...],
  "closing_patterns": ["pattern1", "pattern2", ...],
  "characteristic_phrases": ["phrase1", "phrase2", ...],
  "vocabulary_notes": ["note1", "note2", ...],
  "rhetorical_devices": ["device1", "device2", ...],
  "humor_patterns": ["pattern1", "pattern2", ...],
  "confrontation_style": "description",
  "agreement_style": "description",
  "language_notes": "description of multilingual patterns if any",
  "exemplar_tweets": ["tweet1", "tweet2", ...]
}}
```

## Tweets to Analyze

{tweets}

---

Analyze the voice patterns and return ONLY the JSON object."""


class VoiceExtractor:
    """Extracts voice/style patterns from text corpus."""

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, tweets: list[str], sample_size: int = 200) -> VoiceProfile:
        """
        Extract voice profile from tweets.

        Args:
            tweets: List of tweet texts
            sample_size: How many tweets to sample for analysis (LLM context limit)

        Returns:
            VoiceProfile object
        """
        # Sample tweets evenly across the corpus for representative coverage
        if len(tweets) > sample_size:
            step = len(tweets) // sample_size
            sampled = [tweets[i] for i in range(0, len(tweets), step)][:sample_size]
        else:
            sampled = tweets

        # Format tweets for prompt
        tweets_text = "\n---\n".join(sampled)
        prompt = VOICE_EXTRACTION_PROMPT.format(tweets=tweets_text)

        # Write prompt to temp file
        prompt_file = self.work_dir / "voice_prompt.txt"
        output_file = self.work_dir / "voice_output.json"

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

        # Parse JSON response
        profile_dict = self._parse_json_response(response_text)
        return VoiceProfile.from_dict(profile_dict)

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON object from response."""
        start = text.find('{')
        end = text.rfind('}') + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON object found in response: {text[:500]}...")

        json_str = text[start:end]
        return json.loads(json_str)

    def save_profile(self, profile: VoiceProfile, filepath: str):
        """Save voice profile to JSON file."""
        path = Path(filepath)
        path.write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def load_profile(self, filepath: str) -> VoiceProfile:
        """Load voice profile from JSON file."""
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        return VoiceProfile.from_dict(data)
