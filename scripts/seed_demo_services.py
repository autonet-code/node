#!/usr/bin/env python3
"""Seed a realistic service catalogue into a local daemon, for UX work.

The marketplace design cannot be judged against six listings. This writes
a catalogue with enough breadth that derived topic clusters actually have
something to cluster, and enough depth per topic that a horizontal
category rail has rows worth scrolling.

The categories are NOT invented from a generic app-store taxonomy. They
come from what the services rail can actually support (docs/services_
market.md) plus the market analysis in this repo: a no-reputation,
channel-settled rail selects hard for work that is metered, divisible,
and verifiable-as-you-go. Compute anchors it, because it is the one moat
that cannot be cloned into a free local tool.

Nothing here is declared as a category. Every service is just a name, a
description, an interface and a price; the topics the UI shows are
derived from those embeddings. That is the point: if the clusters that
come out of this look like sensible categories, the derivation works.

Consequently a new entry earns its place in a group by DESCRIPTION, not
by where it sits in this file: write it in the vocabulary its groupmates
use and it will embed next to them. The section comments below are
navigation aids for humans editing this list — the daemon never sees
them.

Images: only the ORIGINAL 27 services carry an ``image_uri``; the newer
entries deliberately have none, so the UI's default colour banner stays
on screen and both paths get exercised in one catalogue.

Usage:
    python scripts/seed_demo_services.py            # add to a running daemon
    python scripts/seed_demo_services.py --clear    # retire existing first
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    import websockets
except ImportError:  # pragma: no cover
    print("pip install websockets", file=sys.stderr)
    raise SystemExit(1)

WS_URL = "ws://127.0.0.1:7700"

# Banner images live in a sibling assets repo, addressed by the kebab-case
# slug of the service name. Only the names in IMAGED below have a file
# there; everything else registers with no image_uri at all.
IMAGE_BASE = "https://raw.githubusercontent.com/autonet-code/assets/main/services"

# (name, description, [param names], ask_atn, provider)
# Providers repeat across services on purpose: a real market has
# multi-service providers, and the "by provider" view needs them.
SERVICES: list[tuple[str, str, list[str], str, str]] = [
    # ---- Compute: the anchor category -------------------------------
    ("A100 GPU Hours",
     "Dedicated NVIDIA A100 80GB, billed per second. Bring your own "
     "container image. Checkpoints stream to your own storage so a "
     "crashed run never costs you the whole job.",
     ["image", "seconds", "checkpoint_uri"], "4", "0xA100c0mputeC0113ct1v3000000000000000001"),
    ("H100 Cluster Time",
     "Eight-way H100 node with NVLink, per-second billing. For training "
     "runs that will not fit on a single card.",
     ["image", "seconds", "nodes"], "31", "0xA100c0mputeC0113ct1v3000000000000000001"),
    ("Consumer GPU Spot",
     "Idle gaming GPUs (4090 / 3090) at spot prices. Cheapest tier, "
     "preemptible, best for batch generation you can retry.",
     ["image", "seconds"], "1", "0xIdl3RigsC00p0000000000000000000000000002"),
    ("Bare Metal VM",
     "Root on a dedicated machine, hourly. No hypervisor tax, no noisy "
     "neighbours.",
     ["image", "hours", "region"], "2", "0xIdl3RigsC00p0000000000000000000000000002"),
    ("Object Storage",
     "S3-compatible buckets billed per GB-month. Random-read audit "
     "endpoint included so you can verify your data is still there.",
     ["bucket", "gb", "months"], "1", "0xC01dSt0r4g3Ltd00000000000000000000000003"),
    ("Archival Cold Storage",
     "Cheap long-term storage with slow retrieval. For data you must "
     "keep but rarely read.",
     ["bucket", "gb", "months"], "1", "0xC01dSt0r4g3Ltd00000000000000000000000003"),
    ("L40S Inference Nodes",
     "Single-card L40S instances sized for serving rather than training, "
     "billed per second. Cold start under thirty seconds, so a bursty "
     "workload is not paying for idle silicon.",
     ["image", "seconds", "region"], "2", "0xA100c0mputeC0113ct1v3000000000000000001"),
    ("CPU Batch Compute",
     "Many-core CPU workers for jobs no GPU helps: compilation, "
     "simulation, format conversion. Billed per core-second, one shard "
     "per work item so a failed shard is re-queued not re-billed.",
     ["image", "core_seconds", "shards"], "1", "0xIdl3RigsC00p0000000000000000000000000002"),
    ("Preemptible Training Queue",
     "Deep-discount GPU capacity that runs when the cluster is quiet. "
     "You supply a checkpoint interval; we bill only the seconds "
     "actually computed before a preemption.",
     ["image", "seconds", "checkpoint_minutes"], "1", "0xIdl3RigsC00p0000000000000000000000000002"),
    ("Block Volume Attach",
     "NVMe block devices attached to your compute node, per GB-hour. "
     "Snapshot on detach so a dataset survives the machine it was "
     "staged on.",
     ["volume", "gb", "hours"], "1", "0xC01dSt0r4g3Ltd00000000000000000000000003"),
    ("Container Registry Hosting",
     "Private image registry billed per GB-month stored plus pulls. "
     "Layer digests published so you can verify the image your run "
     "actually pulled.",
     ["repository", "gb", "months"], "1", "0xN0d3F4br1cInfr40000000000000000000000013"),
    ("Sandboxed Code Execution",
     "Run untrusted code in a locked-down microVM, billed per second. No "
     "network by default; a per-run transcript of syscalls returned with "
     "the result.",
     ["language", "code", "seconds"], "1", "0xN0d3F4br1cInfr40000000000000000000000013"),

    # ---- Inference: the largest by value ----------------------------
    ("Open Model Inference",
     "Llama, Qwen and Mistral served per token. Model fingerprint "
     "published so you can verify you got the weights you paid for.",
     ["model", "messages", "max_tokens"], "1", "0x0p3nW31ghtsH0st000000000000000000000004"),
    ("Frontier Model Relay",
     "Access to top-tier hosted models, per token, no account needed. "
     "Pay per call rather than per month.",
     ["model", "messages", "max_tokens"], "3", "0x0p3nW31ghtsH0st000000000000000000000004"),
    ("Fine-Tuned Legal Model",
     "A model tuned on twenty years of contract law. Specialist weights, "
     "not a system prompt.",
     ["messages", "jurisdiction"], "6", "0xL3g41Sp3c141istM0d3ls00000000000000005"),
    ("Bulk Embeddings",
     "Text to vectors at volume. Sub-cent per thousand, batched, the "
     "staple food of any retrieval pipeline.",
     ["texts", "model"], "1", "0x0p3nW31ghtsH0st000000000000000000000004"),
    ("Reranking Service",
     "Cross-encoder reranking over candidate documents. Cheap enough to "
     "run on every query.",
     ["query", "documents"], "1", "0xR3trievalW0rks00000000000000000000000006"),
    ("Long Context Inference",
     "Served models with a million-token window, priced per token so a "
     "short prompt is not billed at long-context rates. Prompt-cache "
     "hits are discounted and reported per call.",
     ["model", "messages", "max_tokens"], "2", "0x0p3nW31ghtsH0st000000000000000000000004"),
    ("Speculative Decoding Endpoint",
     "The same open weights served with a draft model in front, per "
     "token. Two to three times the throughput at identical outputs, "
     "verified by the target model on every accepted span.",
     ["model", "messages", "max_tokens"], "1", "0x0p3nW31ghtsH0st000000000000000000000004"),
    ("Batch Completion Queue",
     "Overnight inference for work with no latency requirement, per "
     "token at half the interactive rate. Submit a JSONL of prompts, "
     "collect results by morning.",
     ["model", "prompts", "deadline_hours"], "1", "0xBu1kInf3r3nc3W0rks00000000000000000014"),
    ("Fine-Tuning Runs",
     "LoRA and full fine-tunes on open weights, priced per million "
     "training tokens. Loss curve streamed live; the adapter is yours to "
     "download and serve anywhere.",
     ["base_model", "dataset_uri", "tokens"], "9", "0xBu1kInf3r3nc3W0rks00000000000000000014"),
    ("Fine-Tuned Medical Coder",
     "Clinical notes to billing codes, from weights trained on coded "
     "discharge summaries. Specialist model, per document, with the code "
     "set version pinned.",
     ["note", "code_set"], "5", "0xL3g41Sp3c141istM0d3ls00000000000000005"),
    ("Structured Output Enforcement",
     "Constrained decoding against your JSON schema, per call. The model "
     "physically cannot emit an invalid document, so you skip the retry "
     "loop entirely.",
     ["model", "messages", "schema"], "1", "0xBu1kInf3r3nc3W0rks00000000000000000014"),

    # ---- Data and retrieval -----------------------------------------
    ("Web Scraping",
     "Fetch and clean any public page, returned as structured JSON. "
     "Handles JS rendering and pagination.",
     ["url", "selector", "render_js"], "1", "0xScr4p3F4rm00000000000000000000000000007"),
    ("Residential Proxy Pool",
     "Requests routed through residential IPs, per request. For sites "
     "that block datacentre ranges.",
     ["url", "country", "session"], "1", "0xScr4p3F4rm00000000000000000000000000007"),
    ("Market Data Feed",
     "Real-time and historical prices across equities and crypto, per "
     "query.",
     ["symbol", "interval", "from", "to"], "2", "0xF33dsAndT1ck3rs0000000000000000000008"),
    ("Company Filings Index",
     "Full-text search over regulatory filings, per query. Cleaned and "
     "entity-resolved.",
     ["query", "form_type", "since"], "3", "0xF33dsAndT1ck3rs0000000000000000000008"),
    ("Geocoding Lookup",
     "Address to coordinates and back, per lookup. Global coverage.",
     ["address", "country"], "1", "0xR3trievalW0rks00000000000000000000000006"),
    ("Vector Index Hosting",
     "A managed nearest-neighbour index over your embeddings, billed per "
     "thousand queries plus vectors stored. Recall measured against exact "
     "search and reported with every query.",
     ["index", "vector", "top_k"], "1", "0xR3trievalW0rks00000000000000000000000006"),
    ("Search API Passthrough",
     "Web search results as structured JSON, per query. No key to "
     "manage, no monthly floor, and the source URLs come back intact for "
     "citation.",
     ["query", "count", "region"], "1", "0xScr4p3F4rm00000000000000000000000000007"),
    ("Sitemap Crawl",
     "Crawl an entire domain from its sitemap and return cleaned pages, "
     "billed per page fetched. Resumable, so a crawl interrupted at page "
     "nine thousand does not restart.",
     ["domain", "max_pages", "since"], "1", "0xScr4p3F4rm00000000000000000000000000007"),
    ("Corporate Ownership Graph",
     "Who owns whom: parent, subsidiary and beneficial-ownership edges "
     "across registries, per lookup. Each edge cites the filing it came "
     "from.",
     ["entity", "jurisdiction", "depth"], "4", "0xF33dsAndT1ck3rs0000000000000000000008"),
    ("Patent Full-Text Search",
     "Search granted patents and applications across offices, per query. "
     "Claims parsed out separately from the description.",
     ["query", "office", "since"], "2", "0xR3c0rdsAndR3g1str13s000000000000000015"),
    ("Sanctions and PEP Screening",
     "Screen a name against watchlists and politically-exposed-person "
     "records, per check. Every hit returns the list, the entry and the "
     "match score, so a false positive is arguable rather than final.",
     ["name", "country", "date_of_birth"], "2", "0xR3c0rdsAndR3g1str13s000000000000000015"),

    # ---- Media transformation ---------------------------------------
    ("Audio Transcription",
     "Speech to text with speaker labels, billed per minute of audio. "
     "Ninety-eight languages.",
     ["audio_url", "language", "diarize"], "1", "0xM3d14P1p3l1n30000000000000000000000009"),
    ("Text to Speech",
     "Natural voices from text, per thousand characters. Voice cloning "
     "available with consent proof.",
     ["text", "voice", "speed"], "1", "0xM3d14P1p3l1n30000000000000000000000009"),
    ("Image Generation",
     "Text to image, per image. Multiple model families, commercial use "
     "included.",
     ["prompt", "model", "size"], "1", "0xP1x315Unl1m1t3d000000000000000000000010"),
    ("Video Rendering",
     "Render sequences to video, billed per output second. Queue-based "
     "with progress callbacks.",
     ["scene", "seconds", "resolution"], "5", "0xP1x315Unl1m1t3d000000000000000000000010"),
    ("Document OCR",
     "Scanned documents to structured text, per page. Tables and forms "
     "preserved.",
     ["document_url", "language"], "1", "0xM3d14P1p3l1n30000000000000000000000009"),
    ("Video Transcoding",
     "Re-encode video to any codec and ladder, billed per output minute. "
     "Per-segment checksums returned so a partial delivery is provable "
     "rather than disputed.",
     ["video_url", "codec", "resolutions"], "1", "0xP1x315Unl1m1t3d000000000000000000000010"),
    ("Image Upscaling",
     "Restore and enlarge images up to four times, per image. Faces and "
     "text handled by separate passes, so a scanned page stays readable.",
     ["image_url", "scale", "denoise"], "1", "0xP1x315Unl1m1t3d000000000000000000000010"),
    ("Music and Sound Generation",
     "Generate instrumental beds and sound effects from a text brief, "
     "billed per output second. Stems delivered separately, cleared for "
     "commercial use.",
     ["prompt", "seconds", "format"], "2", "0xW4v3f0rmStud10s00000000000000000000016"),
    ("Audio Mastering",
     "Loudness normalisation, de-noising and EQ on a finished mix, per "
     "audio minute. Before-and-after measurements returned with the file.",
     ["audio_url", "target_lufs"], "1", "0xW4v3f0rmStud10s00000000000000000000016"),
    ("Subtitle Translation",
     "Translate and re-time a subtitle track, per audio minute. Timings "
     "adjusted to the target language's reading speed, not just "
     "copied across.",
     ["subtitle_url", "target_language"], "1", "0xW4v3f0rmStud10s00000000000000000000016"),
    ("Handwriting Recognition",
     "Handwritten pages and forms to text, per page. Confidence per field "
     "so a low-scoring line can be routed to a human instead of trusted.",
     ["document_url", "language", "form_template"], "2", "0xM3d14P1p3l1n30000000000000000000000009"),

    # ---- Human in the loop ------------------------------------------
    ("Human Verification",
     "A real person completes a step your agent cannot: a phone call, a "
     "physical signature, a photo taken. Priced per task.",
     ["instructions", "deadline_hours"], "12", "0xHum4nInTh3L00p00000000000000000000011"),
    ("Data Labelling",
     "Human annotation with three-way redundancy and majority vote, per "
     "item.",
     ["items", "schema", "redundancy"], "2", "0xHum4nInTh3L00p00000000000000000000011"),
    ("Content Moderation",
     "Human review of flagged content against your policy, per item, "
     "with written rationale.",
     ["content", "policy"], "2", "0xHum4nInTh3L00p00000000000000000000011"),
    ("Expert Answer Review",
     "A credentialed professional in your field reads a draft and marks "
     "what is wrong, per document. Reviewer credentials disclosed before "
     "you commit the payment.",
     ["document", "field", "deadline_hours"], "18", "0xHum4nInTh3L00p00000000000000000000011"),
    ("Phone Call Errand",
     "A person calls a business on your behalf, follows your script and "
     "returns what was said, per call. Recording attached where the "
     "jurisdiction permits it.",
     ["script", "phone_number", "deadline_hours"], "9", "0xC0nc13rg3T4skF0rc300000000000000000017"),
    ("Local Errand Runner",
     "Someone physically goes somewhere and does one thing: collects a "
     "document, photographs a noticeboard, queues at a counter. Priced "
     "per errand with a photo proof of completion.",
     ["instructions", "city", "deadline_hours"], "22", "0xC0nc13rg3T4skF0rc300000000000000000017"),
    ("Notarised Signature",
     "A notary witnesses and stamps a document, per signature. Scanned "
     "return same day, physical original couriered on request.",
     ["document", "country", "courier"], "35", "0xC0nc13rg3T4skF0rc300000000000000000017"),
    ("Translation Post-Editing",
     "A native-speaking translator corrects machine output, per hundred "
     "words. Tracked changes returned so you can see what the model got "
     "wrong and why.",
     ["text", "source_language", "target_language"], "3", "0xL1ngu4W0rkr00m000000000000000000000018"),
    ("Survey Panel Responses",
     "Recruit screened respondents and collect answers to your "
     "questionnaire, per completed response. Attention checks included "
     "and failed responses are not billed.",
     ["questions", "screening", "responses"], "4", "0xL1ngu4W0rkr00m000000000000000000000018"),
    ("Dataset Audit",
     "A human samples your training data and reports label error rate, "
     "leakage and bias, per hundred rows sampled. Findings itemised, not "
     "scored.",
     ["dataset_uri", "sample_size", "schema"], "6", "0xHum4nInTh3L00p00000000000000000000011"),

    # ---- Agent operations -------------------------------------------
    ("Persistent Agent Hosting",
     "Keep an agent running while you sleep. Billed per uptime hour, "
     "with a public endpoint and restart-on-crash.",
     ["agent_image", "hours"], "2", "0xAlw4ys0nH0st1ng000000000000000000000012"),
    ("Scheduled Triggers",
     "Cron for agents: fire a webhook on a schedule you set, per "
     "trigger.",
     ["schedule", "webhook", "payload"], "1", "0xAlw4ys0nH0st1ng000000000000000000000012"),
    ("Uptime Monitoring",
     "Poll an endpoint and alert on failure, per check. Verifiable "
     "because you can poll it yourself.",
     ["url", "interval_seconds"], "1", "0xAlw4ys0nH0st1ng000000000000000000000012"),
    ("Managed Agent Memory",
     "Durable conversation and task memory for an agent that outlives its "
     "process, billed per thousand reads and writes. Full export on "
     "demand, so leaving costs nothing but a download.",
     ["agent_id", "operation", "payload"], "1", "0xAlw4ys0nH0st1ng000000000000000000000012"),
    ("Webhook Relay",
     "A stable public URL that forwards inbound events to an agent behind "
     "a home connection, per delivered event. Retries with backoff and "
     "replay of the last day of traffic.",
     ["target", "secret", "retries"], "1", "0xR3l4yAndR0ut30000000000000000000000019"),
    ("Headless Browser Sessions",
     "A real browser an agent can drive, billed per session minute. "
     "Logged-in state persists between sessions and screenshots come back "
     "with every step.",
     ["start_url", "minutes", "profile"], "2", "0xR3l4yAndR0ut30000000000000000000000019"),
    ("Outbound Email Delivery",
     "Send mail from your own domain with authentication configured, per "
     "message. Bounce and complaint events posted back so an agent knows "
     "what actually landed.",
     ["to", "subject", "body"], "1", "0xR3l4yAndR0ut30000000000000000000000019"),
    ("SMS and Voice Numbers",
     "Rent a phone number an agent can send from and receive on, billed "
     "per message and per call minute. Inbound arrives as a webhook.",
     ["number", "message", "country"], "1", "0xR3l4yAndR0ut30000000000000000000000019"),
    ("Log Retention and Search",
     "Ship an agent's run logs somewhere durable and query them, billed "
     "per GB ingested. Thirty-day retention, full-text search over the "
     "whole window.",
     ["stream", "gb", "query"], "1", "0x0bs3rv3P14tf0rm00000000000000000000020"),
    ("Agent Trace Evaluation",
     "Score an agent's completed runs against a rubric you supply, per "
     "trace. Per-step verdicts with the failing step named, not a single "
     "opaque number.",
     ["trace", "rubric", "model"], "2", "0x0bs3rv3P14tf0rm00000000000000000000020"),
    ("Secret Vault Access",
     "Hold an agent's credentials outside its own process and lease them "
     "per request. Every access is logged by name, and a lease expires "
     "whether the agent releases it or not.",
     ["secret_name", "ttl_seconds", "reason"], "1", "0x0bs3rv3P14tf0rm00000000000000000000020"),
]

# The 27 services that shipped in the first cut of this catalogue: exactly
# the names with a banner file in the assets repo. Newer entries are left
# imageless on purpose (see module docstring) — adding one here without
# adding the file just yields a broken image.
IMAGED: frozenset[str] = frozenset({
    "A100 GPU Hours", "H100 Cluster Time", "Consumer GPU Spot",
    "Bare Metal VM", "Object Storage", "Archival Cold Storage",
    "Open Model Inference", "Frontier Model Relay", "Fine-Tuned Legal Model",
    "Bulk Embeddings", "Reranking Service",
    "Web Scraping", "Residential Proxy Pool", "Market Data Feed",
    "Company Filings Index", "Geocoding Lookup",
    "Audio Transcription", "Text to Speech", "Image Generation",
    "Video Rendering", "Document OCR",
    "Human Verification", "Data Labelling", "Content Moderation",
    "Persistent Agent Hosting", "Scheduled Triggers", "Uptime Monitoring",
})


def _slug(name: str) -> str:
    """Kebab-case slug of a service name, matching the asset filenames."""
    return "-".join("".join(
        c if (c.isalnum() or c.isspace()) else " " for c in name.lower()
    ).split())


def _image_uri(name: str) -> str:
    """Banner URL for the imaged originals; empty for everything else."""
    return f"{IMAGE_BASE}/{_slug(name)}.jpg" if name in IMAGED else ""


def _spec(name: str, description: str, params: list[str], ask: str,
          author: str) -> dict:
    msg = {
        "type": "register_service",
        "name": name,
        "description": description,
        "agent_id": author,
        "backing_tool": "",
        "input_schema": {
            "type": "object",
            "properties": {p: {"type": "string"} for p in params},
            "required": params[:1],
        },
        # ATN-denominated by construction (ratified 2026-07-10) — no
        # token field.
        "ask": {"amount": ask, "unit": "per_item"},
    }
    # Omitted rather than sent empty for the imageless entries: the field
    # is optional on the wire, and an absent key is the honest signal.
    image_uri = _image_uri(name)
    if image_uri:
        msg["image_uri"] = image_uri
    return msg


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true",
                        help="retire every existing service first")
    parser.add_argument("--url", default=WS_URL)
    args = parser.parse_args()

    async with websockets.connect(args.url, max_size=None) as ws:
        async def call(payload: dict) -> dict:
            mid = f"seed-{id(payload)}"
            await ws.send(json.dumps({**payload, "msg_id": mid}))
            while True:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if resp.get("msg_id") == mid:
                    return resp

        if args.clear:
            existing = await call({"type": "list_services"})
            for svc in (existing.get("result") or {}).get("services", []):
                await call({"type": "retire_service",
                            "digest": svc.get("digest", "")})
            print(f"retired {len((existing.get('result') or {}).get('services', []))}")

        ok = 0
        for name, desc, params, ask, author in SERVICES:
            resp = await call(_spec(name, desc, params, ask, author))
            if resp.get("ok"):
                ok += 1
            else:
                print(f"  FAILED {name}: {resp.get('error')}", file=sys.stderr)
        imaged = sum(1 for s in SERVICES if _image_uri(s[0]))
        print(f"registered {ok}/{len(SERVICES)} services "
              f"across {len({s[4] for s in SERVICES})} providers "
              f"({imaged} with a banner image, {len(SERVICES) - imaged} without)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
