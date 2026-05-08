"""
Processes Twitter archive for World Model extraction and voice pattern analysis.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tweet:
    """A processed tweet."""
    content: str
    is_reply: bool = False
    is_retweet: bool = False
    has_link: bool = False
    has_media: bool = False
    language: str = "en"  # or "ro" for Romanian
    tags: list[str] = field(default_factory=list)


class TweetProcessor:
    """Processes raw tweet text into structured data for extraction."""

    # Romanian indicators
    RO_WORDS = {
        'și', 'că', 'nu', 'în', 'să', 'de', 'la', 'pe', 'cu', 'ce',
        'dar', 'sau', 'mi', 'te', 'eu', 'tu', 'el', 'ea', 'noi',
        'dacă', 'când', 'unde', 'cum', 'pentru', 'despre', 'prin',
        'ăsta', 'asta', 'ești', 'sunt', 'este', 'eram', 'fac', 'zice',
        'foarte', 'mai', 'doar', 'încă', 'deja', 'mereu', 'toți',
    }

    # Link pattern
    LINK_PATTERN = re.compile(r'https?://\S+')

    # Auto-generated marker
    AUTO_PATTERN = re.compile(r'^\[auto\]\s*', re.IGNORECASE)

    def __init__(self):
        self.stats = {
            'total': 0,
            'filtered_reply_only': 0,
            'filtered_link_only': 0,
            'filtered_too_short': 0,
            'english': 0,
            'romanian': 0,
            'substantive': 0,
        }

    def process_file(self, filepath: str) -> list[Tweet]:
        """Process tweets_text_only.txt format."""
        path = Path(filepath)
        content = path.read_text(encoding='utf-8')

        # Split by double newlines (tweet separator)
        raw_tweets = [t.strip() for t in content.split('\n\n') if t.strip()]

        tweets = []
        for raw in raw_tweets:
            self.stats['total'] += 1
            tweet = self._process_tweet(raw)
            if tweet:
                tweets.append(tweet)

        return tweets

    def _process_tweet(self, raw: str) -> Tweet | None:
        """Process a single tweet, return None if filtered out."""
        content = raw.strip()

        # Remove [auto] prefix but keep the content
        content = self.AUTO_PATTERN.sub('', content)

        # Check if reply-only (starts with @ and has no other content)
        if content.startswith('@'):
            # Extract non-mention content
            words = content.split()
            non_mention_words = [w for w in words if not w.startswith('@')]
            if len(non_mention_words) < 3:
                self.stats['filtered_reply_only'] += 1
                return None
            # Keep it but mark as reply
            is_reply = True
        else:
            is_reply = False

        # Remove links for content analysis
        content_no_links = self.LINK_PATTERN.sub('', content).strip()

        # Check if link-only
        if len(content_no_links) < 10:
            self.stats['filtered_link_only'] += 1
            return None

        # Check minimum substance
        if len(content_no_links.split()) < 4:
            self.stats['filtered_too_short'] += 1
            return None

        # Detect language
        words = set(content_no_links.lower().split())
        ro_count = len(words & self.RO_WORDS)
        language = 'ro' if ro_count >= 2 else 'en'

        if language == 'ro':
            self.stats['romanian'] += 1
        else:
            self.stats['english'] += 1

        self.stats['substantive'] += 1

        return Tweet(
            content=content,
            is_reply=is_reply,
            has_link=bool(self.LINK_PATTERN.search(content)),
            language=language,
        )

    def get_substantive_tweets(self, tweets: list[Tweet], language: str | None = None) -> list[str]:
        """Get tweet contents, optionally filtered by language."""
        result = []
        for t in tweets:
            if language and t.language != language:
                continue
            result.append(t.content)
        return result

    def batch_for_extraction(self, tweets: list[str], batch_size: int = 50) -> list[str]:
        """
        Group tweets into batches for LLM extraction.
        Returns list of batch strings.
        """
        batches = []
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i + batch_size]
            batch_text = "\n---\n".join(batch)
            batches.append(batch_text)
        return batches


def main():
    """Test the processor."""
    processor = TweetProcessor()
    tweets = processor.process_file(
        r"C:\Users\astmo\Downloads\twitter archive\tweets_text_only.txt"
    )

    print(f"Processing stats:")
    for key, val in processor.stats.items():
        print(f"  {key}: {val}")

    print(f"\nSubstantive tweets: {len(tweets)}")

    # Sample
    print("\n--- Sample English tweets ---")
    en_tweets = [t for t in tweets if t.language == 'en'][:5]
    for t in en_tweets:
        print(f"  {t.content[:80]}...")

    print("\n--- Sample Romanian tweets ---")
    ro_tweets = [t for t in tweets if t.language == 'ro'][:5]
    for t in ro_tweets:
        print(f"  {t.content[:80]}...")


if __name__ == "__main__":
    main()
