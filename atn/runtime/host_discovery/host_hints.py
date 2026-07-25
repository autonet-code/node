"""Curated secret-name → authorized-egress-host hints.

A convenience prefill only: when the owner stores a secret whose name matches a
well-known pattern, we can suggest the host(s) that secret is legitimately used
against, so the owner confirms rather than types. NOT a security boundary — the
owner may edit or clear the suggestion, and an unmatched name simply yields no
suggestion (the owner must supply hosts explicitly to enable proxy mode).

Match is case-insensitive substring on the secret NAME (e.g. any name
containing "OPENAI" suggests api.openai.com). Order matters: the first entry
whose key is a substring of the name wins, so keep more specific keys first.
"""
from __future__ import annotations

# (name-substring, [authorized hosts]). First substring match wins.
_HINTS: list[tuple[str, list[str]]] = [
    ("ANTHROPIC", ["api.anthropic.com"]),
    ("OPENAI", ["api.openai.com"]),
    ("DEEPSEEK", ["api.deepseek.com"]),
    ("GEMINI", ["generativelanguage.googleapis.com"]),
    ("GOOGLE_API", ["generativelanguage.googleapis.com"]),
    ("ELEVENLABS", ["api.elevenlabs.io"]),
    ("ELEVEN_LABS", ["api.elevenlabs.io"]),
    ("MISTRAL", ["api.mistral.ai"]),
    ("GROQ", ["api.groq.com"]),
    ("COHERE", ["api.cohere.com"]),
    ("PERPLEXITY", ["api.perplexity.ai"]),
    ("HUGGINGFACE", ["huggingface.co", "api-inference.huggingface.co"]),
    ("HF_TOKEN", ["huggingface.co", "api-inference.huggingface.co"]),
    ("REPLICATE", ["api.replicate.com"]),
    ("OPENROUTER", ["openrouter.ai"]),
    ("TOGETHER", ["api.together.xyz"]),
    ("GITHUB", ["api.github.com"]),
    ("GITLAB", ["gitlab.com"]),
    ("STRIPE", ["api.stripe.com"]),
    ("SENDGRID", ["api.sendgrid.com"]),
    ("TWILIO", ["api.twilio.com"]),
    ("SLACK", ["slack.com"]),
    ("DISCORD", ["discord.com"]),
    ("TELEGRAM", ["api.telegram.org"]),
    ("NOTION", ["api.notion.com"]),
    ("LINEAR", ["api.linear.app"]),
    ("AIRTABLE", ["api.airtable.com"]),
    ("SUPABASE", ["supabase.co"]),
    ("PINECONE", ["pinecone.io"]),
    ("TAVILY", ["api.tavily.com"]),
    ("SERPAPI", ["serpapi.com"]),
    ("BRAVE", ["api.search.brave.com"]),
    ("ETHERSCAN", ["api.etherscan.io"]),
    ("ALCHEMY", ["alchemy.com"]),
    ("INFURA", ["infura.io"]),
    ("COINGECKO", ["api.coingecko.com"]),
    ("CLOUDFLARE", ["api.cloudflare.com"]),
    ("DIGITALOCEAN", ["api.digitalocean.com"]),
    ("VERCEL", ["api.vercel.com"]),
]


def suggest_hosts(secret_name: str) -> list[str]:
    """Suggested authorized hosts for a secret name, or [] if none matched."""
    if not secret_name:
        return []
    upper = secret_name.upper()
    for key, hosts in _HINTS:
        if key in upper:
            return list(hosts)
    return []
