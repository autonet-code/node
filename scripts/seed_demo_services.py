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

Working listings, on a local hardhat jurisdiction: every listing is
registered on chain in the jurisdiction's ServiceRegistry, priced in ATN
(converted to base units, 1 ATN = 1e6), and backed by one demo tool this
script registers, which answers each purchase with the arguments it got
and says it is a demo. The daemon must have joined a local network with a
ServiceRegistry (``scripts/manual_local_net.py`` or ``s0_local_run.py``),
and the address that registers must be live on Substrate: by default the
owner surface, which registers under the operator address (register the
operator in the app's Household first); with ``--agent`` an on-chain
registered agent whose key the daemon holds. ``--host`` seeds only the
listings of the named provider houses (the labels below), so a catalogue
with several providers is several daemons, one house set each.

Usage:
    python scripts/seed_demo_services.py              # into the daemon on :7700
    python scripts/seed_demo_services.py --clear      # retire existing first
    python scripts/seed_demo_services.py --url ws://127.0.0.1:7710 \\
        --agent provider-1 --host open-weights --host bulk-inference
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal

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

# ATN has 6 decimals: an ask of 1 ATN is 1_000_000 base units on chain.
ATN_BASE_UNITS = 10 ** 6

# (name, description, [param names], ask in ATN, provider house)
# Houses repeat across services on purpose: a real market has
# multi-service providers, and the "by provider" view needs them. A house
# is only a label here: the provider of record is the address that
# registers the listing (see the module docstring, --host).
SERVICES: list[tuple[str, str, list[str], str, str]] = [
    # ---- Compute: the anchor category -------------------------------
    ("A100 GPU Hours",
     "Dedicated NVIDIA A100 80GB, billed per second. Bring your own "
     "container image. Checkpoints stream to your own storage so a "
     "crashed run never costs you the whole job.",
     ["image", "seconds", "checkpoint_uri"], "4", "compute-collective"),
    ("H100 Cluster Time",
     "Eight-way H100 node with NVLink, per-second billing. For training "
     "runs that will not fit on a single card.",
     ["image", "seconds", "nodes"], "31", "compute-collective"),
    ("Consumer GPU Spot",
     "Idle gaming GPUs (4090 / 3090) at spot prices. Cheapest tier, "
     "preemptible, best for batch generation you can retry.",
     ["image", "seconds"], "1", "idle-rigs"),
    ("Bare Metal VM",
     "Root on a dedicated machine, hourly. No hypervisor tax, no noisy "
     "neighbours.",
     ["image", "hours", "region"], "2", "idle-rigs"),
    ("Object Storage",
     "S3-compatible buckets billed per GB-month. Random-read audit "
     "endpoint included so you can verify your data is still there.",
     ["bucket", "gb", "months"], "1", "cold-storage"),
    ("Archival Cold Storage",
     "Cheap long-term storage with slow retrieval. For data you must "
     "keep but rarely read.",
     ["bucket", "gb", "months"], "1", "cold-storage"),
    ("L40S Inference Nodes",
     "Single-card L40S instances sized for serving rather than training, "
     "billed per second. Cold start under thirty seconds, so a bursty "
     "workload is not paying for idle silicon.",
     ["image", "seconds", "region"], "2", "compute-collective"),
    ("CPU Batch Compute",
     "Many-core CPU workers for jobs no GPU helps: compilation, "
     "simulation, format conversion. Billed per core-second, one shard "
     "per work item so a failed shard is re-queued not re-billed.",
     ["image", "core_seconds", "shards"], "1", "idle-rigs"),
    ("Preemptible Training Queue",
     "Deep-discount GPU capacity that runs when the cluster is quiet. "
     "You supply a checkpoint interval; we bill only the seconds "
     "actually computed before a preemption.",
     ["image", "seconds", "checkpoint_minutes"], "1", "idle-rigs"),
    ("Block Volume Attach",
     "NVMe block devices attached to your compute node, per GB-hour. "
     "Snapshot on detach so a dataset survives the machine it was "
     "staged on.",
     ["volume", "gb", "hours"], "1", "cold-storage"),
    ("Container Registry Hosting",
     "Private image registry billed per GB-month stored plus pulls. "
     "Layer digests published so you can verify the image your run "
     "actually pulled.",
     ["repository", "gb", "months"], "1", "node-fabric"),
    ("Sandboxed Code Execution",
     "Run untrusted code in a locked-down microVM, billed per second. No "
     "network by default; a per-run transcript of syscalls returned with "
     "the result.",
     ["language", "code", "seconds"], "1", "node-fabric"),

    # ---- Inference: the largest by value ----------------------------
    ("Open Model Inference",
     "Llama, Qwen and Mistral served per token. Model fingerprint "
     "published so you can verify you got the weights you paid for.",
     ["model", "messages", "max_tokens"], "1", "open-weights"),
    ("Frontier Model Relay",
     "Access to top-tier hosted models, per token, no account needed. "
     "Pay per call rather than per month.",
     ["model", "messages", "max_tokens"], "3", "open-weights"),
    ("Fine-Tuned Legal Model",
     "A model tuned on twenty years of contract law. Specialist weights, "
     "not a system prompt.",
     ["messages", "jurisdiction"], "6", "specialist-models"),
    ("Bulk Embeddings",
     "Text to vectors at volume. Sub-cent per thousand, batched, the "
     "staple food of any retrieval pipeline.",
     ["texts", "model"], "1", "open-weights"),
    ("Reranking Service",
     "Cross-encoder reranking over candidate documents. Cheap enough to "
     "run on every query.",
     ["query", "documents"], "1", "retrieval-works"),
    ("Long Context Inference",
     "Served models with a million-token window, priced per token so a "
     "short prompt is not billed at long-context rates. Prompt-cache "
     "hits are discounted and reported per call.",
     ["model", "messages", "max_tokens"], "2", "open-weights"),
    ("Speculative Decoding Endpoint",
     "The same open weights served with a draft model in front, per "
     "token. Two to three times the throughput at identical outputs, "
     "verified by the target model on every accepted span.",
     ["model", "messages", "max_tokens"], "1", "open-weights"),
    ("Batch Completion Queue",
     "Overnight inference for work with no latency requirement, per "
     "token at half the interactive rate. Submit a JSONL of prompts, "
     "collect results by morning.",
     ["model", "prompts", "deadline_hours"], "1", "bulk-inference"),
    ("Fine-Tuning Runs",
     "LoRA and full fine-tunes on open weights, priced per million "
     "training tokens. Loss curve streamed live; the adapter is yours to "
     "download and serve anywhere.",
     ["base_model", "dataset_uri", "tokens"], "9", "bulk-inference"),
    ("Fine-Tuned Medical Coder",
     "Clinical notes to billing codes, from weights trained on coded "
     "discharge summaries. Specialist model, per document, with the code "
     "set version pinned.",
     ["note", "code_set"], "5", "specialist-models"),
    ("Structured Output Enforcement",
     "Constrained decoding against your JSON schema, per call. The model "
     "physically cannot emit an invalid document, so you skip the retry "
     "loop entirely.",
     ["model", "messages", "schema"], "1", "bulk-inference"),

    # ---- Data and retrieval -----------------------------------------
    ("Web Scraping",
     "Fetch and clean any public page, returned as structured JSON. "
     "Handles JS rendering and pagination.",
     ["url", "selector", "render_js"], "1", "scrape-farm"),
    ("Residential Proxy Pool",
     "Requests routed through residential IPs, per request. For sites "
     "that block datacentre ranges.",
     ["url", "country", "session"], "1", "scrape-farm"),
    ("Market Data Feed",
     "Real-time and historical prices across equities and crypto, per "
     "query.",
     ["symbol", "interval", "from", "to"], "2", "feeds-and-tickers"),
    ("Company Filings Index",
     "Full-text search over regulatory filings, per query. Cleaned and "
     "entity-resolved.",
     ["query", "form_type", "since"], "3", "feeds-and-tickers"),
    ("Geocoding Lookup",
     "Address to coordinates and back, per lookup. Global coverage.",
     ["address", "country"], "1", "retrieval-works"),
    ("Vector Index Hosting",
     "A managed nearest-neighbour index over your embeddings, billed per "
     "thousand queries plus vectors stored. Recall measured against exact "
     "search and reported with every query.",
     ["index", "vector", "top_k"], "1", "retrieval-works"),
    ("Search API Passthrough",
     "Web search results as structured JSON, per query. No key to "
     "manage, no monthly floor, and the source URLs come back intact for "
     "citation.",
     ["query", "count", "region"], "1", "scrape-farm"),
    ("Sitemap Crawl",
     "Crawl an entire domain from its sitemap and return cleaned pages, "
     "billed per page fetched. Resumable, so a crawl interrupted at page "
     "nine thousand does not restart.",
     ["domain", "max_pages", "since"], "1", "scrape-farm"),
    ("Corporate Ownership Graph",
     "Who owns whom: parent, subsidiary and beneficial-ownership edges "
     "across registries, per lookup. Each edge cites the filing it came "
     "from.",
     ["entity", "jurisdiction", "depth"], "4", "feeds-and-tickers"),
    ("Patent Full-Text Search",
     "Search granted patents and applications across offices, per query. "
     "Claims parsed out separately from the description.",
     ["query", "office", "since"], "2", "records-registries"),
    ("Sanctions and PEP Screening",
     "Screen a name against watchlists and politically-exposed-person "
     "records, per check. Every hit returns the list, the entry and the "
     "match score, so a false positive is arguable rather than final.",
     ["name", "country", "date_of_birth"], "2", "records-registries"),

    # ---- Media transformation ---------------------------------------
    ("Audio Transcription",
     "Speech to text with speaker labels, billed per minute of audio. "
     "Ninety-eight languages.",
     ["audio_url", "language", "diarize"], "1", "media-pipeline"),
    ("Text to Speech",
     "Natural voices from text, per thousand characters. Voice cloning "
     "available with consent proof.",
     ["text", "voice", "speed"], "1", "media-pipeline"),
    ("Image Generation",
     "Text to image, per image. Multiple model families, commercial use "
     "included.",
     ["prompt", "model", "size"], "1", "pixels-unlimited"),
    ("Video Rendering",
     "Render sequences to video, billed per output second. Queue-based "
     "with progress callbacks.",
     ["scene", "seconds", "resolution"], "5", "pixels-unlimited"),
    ("Document OCR",
     "Scanned documents to structured text, per page. Tables and forms "
     "preserved.",
     ["document_url", "language"], "1", "media-pipeline"),
    ("Video Transcoding",
     "Re-encode video to any codec and ladder, billed per output minute. "
     "Per-segment checksums returned so a partial delivery is provable "
     "rather than disputed.",
     ["video_url", "codec", "resolutions"], "1", "pixels-unlimited"),
    ("Image Upscaling",
     "Restore and enlarge images up to four times, per image. Faces and "
     "text handled by separate passes, so a scanned page stays readable.",
     ["image_url", "scale", "denoise"], "1", "pixels-unlimited"),
    ("Music and Sound Generation",
     "Generate instrumental beds and sound effects from a text brief, "
     "billed per output second. Stems delivered separately, cleared for "
     "commercial use.",
     ["prompt", "seconds", "format"], "2", "waveform-studios"),
    ("Audio Mastering",
     "Loudness normalisation, de-noising and EQ on a finished mix, per "
     "audio minute. Before-and-after measurements returned with the file.",
     ["audio_url", "target_lufs"], "1", "waveform-studios"),
    ("Subtitle Translation",
     "Translate and re-time a subtitle track, per audio minute. Timings "
     "adjusted to the target language's reading speed, not just "
     "copied across.",
     ["subtitle_url", "target_language"], "1", "waveform-studios"),
    ("Handwriting Recognition",
     "Handwritten pages and forms to text, per page. Confidence per field "
     "so a low-scoring line can be routed to a human instead of trusted.",
     ["document_url", "language", "form_template"], "2", "media-pipeline"),

    # ---- Human in the loop ------------------------------------------
    ("Human Verification",
     "A real person completes a step your agent cannot: a phone call, a "
     "physical signature, a photo taken. Priced per task.",
     ["instructions", "deadline_hours"], "12", "human-in-the-loop"),
    ("Data Labelling",
     "Human annotation with three-way redundancy and majority vote, per "
     "item.",
     ["items", "schema", "redundancy"], "2", "human-in-the-loop"),
    ("Content Moderation",
     "Human review of flagged content against your policy, per item, "
     "with written rationale.",
     ["content", "policy"], "2", "human-in-the-loop"),
    ("Expert Answer Review",
     "A credentialed professional in your field reads a draft and marks "
     "what is wrong, per document. Reviewer credentials disclosed before "
     "you commit the payment.",
     ["document", "field", "deadline_hours"], "18", "human-in-the-loop"),
    ("Phone Call Errand",
     "A person calls a business on your behalf, follows your script and "
     "returns what was said, per call. Recording attached where the "
     "jurisdiction permits it.",
     ["script", "phone_number", "deadline_hours"], "9", "concierge-taskforce"),
    ("Local Errand Runner",
     "Someone physically goes somewhere and does one thing: collects a "
     "document, photographs a noticeboard, queues at a counter. Priced "
     "per errand with a photo proof of completion.",
     ["instructions", "city", "deadline_hours"], "22", "concierge-taskforce"),
    ("Notarised Signature",
     "A notary witnesses and stamps a document, per signature. Scanned "
     "return same day, physical original couriered on request.",
     ["document", "country", "courier"], "35", "concierge-taskforce"),
    ("Translation Post-Editing",
     "A native-speaking translator corrects machine output, per hundred "
     "words. Tracked changes returned so you can see what the model got "
     "wrong and why.",
     ["text", "source_language", "target_language"], "3", "lingua-workroom"),
    ("Survey Panel Responses",
     "Recruit screened respondents and collect answers to your "
     "questionnaire, per completed response. Attention checks included "
     "and failed responses are not billed.",
     ["questions", "screening", "responses"], "4", "lingua-workroom"),
    ("Dataset Audit",
     "A human samples your training data and reports label error rate, "
     "leakage and bias, per hundred rows sampled. Findings itemised, not "
     "scored.",
     ["dataset_uri", "sample_size", "schema"], "6", "human-in-the-loop"),

    # ---- Agent operations -------------------------------------------
    ("Persistent Agent Hosting",
     "Keep an agent running while you sleep. Billed per uptime hour, "
     "with a public endpoint and restart-on-crash.",
     ["agent_image", "hours"], "2", "always-on-hosting"),
    ("Scheduled Triggers",
     "Cron for agents: fire a webhook on a schedule you set, per "
     "trigger.",
     ["schedule", "webhook", "payload"], "1", "always-on-hosting"),
    ("Uptime Monitoring",
     "Poll an endpoint and alert on failure, per check. Verifiable "
     "because you can poll it yourself.",
     ["url", "interval_seconds"], "1", "always-on-hosting"),
    ("Managed Agent Memory",
     "Durable conversation and task memory for an agent that outlives its "
     "process, billed per thousand reads and writes. Full export on "
     "demand, so leaving costs nothing but a download.",
     ["agent_id", "operation", "payload"], "1", "always-on-hosting"),
    ("Webhook Relay",
     "A stable public URL that forwards inbound events to an agent behind "
     "a home connection, per delivered event. Retries with backoff and "
     "replay of the last day of traffic.",
     ["target", "secret", "retries"], "1", "relay-and-route"),
    ("Headless Browser Sessions",
     "A real browser an agent can drive, billed per session minute. "
     "Logged-in state persists between sessions and screenshots come back "
     "with every step.",
     ["start_url", "minutes", "profile"], "2", "relay-and-route"),
    ("Outbound Email Delivery",
     "Send mail from your own domain with authentication configured, per "
     "message. Bounce and complaint events posted back so an agent knows "
     "what actually landed.",
     ["to", "subject", "body"], "1", "relay-and-route"),
    ("SMS and Voice Numbers",
     "Rent a phone number an agent can send from and receive on, billed "
     "per message and per call minute. Inbound arrives as a webhook.",
     ["number", "message", "country"], "1", "relay-and-route"),
    ("Log Retention and Search",
     "Ship an agent's run logs somewhere durable and query them, billed "
     "per GB ingested. Thirty-day retention, full-text search over the "
     "whole window.",
     ["stream", "gb", "query"], "1", "observe-platform"),
    ("Agent Trace Evaluation",
     "Score an agent's completed runs against a rubric you supply, per "
     "trace. Per-step verdicts with the failing step named, not a single "
     "opaque number.",
     ["trace", "rubric", "model"], "2", "observe-platform"),
    ("Secret Vault Access",
     "Hold an agent's credentials outside its own process and lease them "
     "per request. Every access is logged by name, and a lease expires "
     "whether the agent releases it or not.",
     ["secret_name", "ttl_seconds", "reason"], "1", "observe-platform"),
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


def ask_base_units(ask_atn: str) -> int:
    """An ask written in ATN, as the integer base units the chain holds."""
    units = Decimal(ask_atn) * ATN_BASE_UNITS
    if units != units.to_integral_value() or units < 0:
        raise ValueError(f"ask {ask_atn} ATN is not a whole number of base units")
    return int(units)


# The one backing every demo listing dispatches to: a purchase returns what
# was sent and says plainly it came from a demo listing.
DEMO_TOOL_NAME = "demo_service_echo"
DEMO_TOOL_CODE = (
    "import sys, json\n"
    "args = json.load(sys.stdin)\n"
    "print(json.dumps({'demo': True, 'received': args,\n"
    "                  'note': 'demo listing from scripts/seed_demo_services.py: '\n"
    "                          'no real work is done'}))\n"
)


def _spec(name: str, description: str, params: list[str], ask: str,
          backing_tool: str, agent: str = "") -> dict:
    msg = {
        "type": "register_service",
        "name": name,
        "description": description,
        "backing_tool": backing_tool,
        "input_schema": {
            "type": "object",
            "properties": {p: {"type": "string"} for p in params},
            "required": params[:1],
        },
        # The chain's ask is in ATN base units (the spec carries no price).
        "ask": {"amount": str(ask_base_units(ask)), "unit": "per_item"},
        # What one work item is (required, Decision 2026-11-10).
        "unit": "one request",
    }
    if agent:
        msg["agent_id"] = agent
    # Omitted rather than sent empty for the imageless entries: the field
    # is optional on the wire, and an absent key is the honest signal.
    image_uri = _image_uri(name)
    if image_uri:
        msg["image_uri"] = image_uri
    return msg


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed demo service listings into a daemon that joined a "
                    "local hardhat jurisdiction.")
    parser.add_argument("--clear", action="store_true",
                        help="retire every existing service first")
    parser.add_argument("--url", default=WS_URL,
                        help="the daemon's local WebSocket (default %(default)s)")
    parser.add_argument("--agent", default="",
                        help="register as this on-chain agent instead of the "
                             "owner surface (the operator address)")
    parser.add_argument("--host", action="append", default=[],
                        help="seed only this provider house (repeatable)")
    args = parser.parse_args()

    houses = {s[4] for s in SERVICES}
    unknown = [h for h in args.host if h not in houses]
    if unknown:
        print(f"unknown house(s) {unknown}; known: {sorted(houses)}",
              file=sys.stderr)
        return 2
    chosen = [s for s in SERVICES if not args.host or s[4] in args.host]

    async with websockets.connect(args.url, max_size=None) as ws:
        seq = 0

        async def call(payload: dict) -> dict:
            nonlocal seq
            seq += 1
            mid = f"seed-{seq}"
            await ws.send(json.dumps({**payload, "msg_id": mid}))
            while True:
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                if resp.get("msg_id") == mid:
                    return resp

        if not args.agent:
            # The owner surface registers under the operator: it must be
            # live on Substrate, or every listing stays local-only.
            status = await call({"type": "operator_status"})
            st = status.get("result") or {}
            if not status.get("ok") or st.get("error"):
                print(f"this daemon is on no chain with a Substrate: "
                      f"{status.get('error') or st.get('error')}",
                      file=sys.stderr)
                return 1
            if not st.get("live"):
                print(f"the operator {st.get('operator')} is not registered "
                      "and live on Substrate: register it (the app's "
                      "Household) or pass --agent", file=sys.stderr)
                return 1

        if args.clear:
            existing = await call({"type": "list_services"})
            for svc in (existing.get("result") or {}).get("services", []):
                await call({"type": "retire_service",
                            "digest": svc.get("digest", "")})
            print(f"retired {len((existing.get('result') or {}).get('services', []))}")

        # Registered by the owner whoever sells: a service's backing runs
        # with the owner's sanction (the sale is the owner's choice), and
        # agents hold register_tool only by grant.
        tool = await call({"type": "register_tool", "name": DEMO_TOOL_NAME,
                           "description": "Echoes a demo service purchase.",
                           "input_schema": {"type": "object"},
                           "code": DEMO_TOOL_CODE})
        backing = (tool.get("result") or {}).get("digest", "")
        if not tool.get("ok") or not backing:
            print(f"could not register the demo backing tool: "
                  f"{tool.get('error') or tool}", file=sys.stderr)
            return 1

        on_chain = local_only = failed = 0
        for name, desc, params, ask, _house in chosen:
            resp = await call(_spec(name, desc, params, ask, backing,
                                    args.agent))
            result = resp.get("result") or {}
            if not resp.get("ok"):
                failed += 1
                print(f"  FAILED {name}: {resp.get('error')}", file=sys.stderr)
            elif result.get("on_chain"):
                on_chain += 1
            else:
                local_only += 1
                print(f"  LOCAL ONLY {name}: {result.get('reason')}",
                      file=sys.stderr)
        imaged = sum(1 for s in chosen if _image_uri(s[0]))
        print(f"listed {on_chain}/{len(chosen)} services on chain "
              f"({local_only} local only, {failed} failed) from "
              f"{len({s[4] for s in chosen})} houses, backed by "
              f"{DEMO_TOOL_NAME} {backing[:16]} "
              f"({imaged} with a banner image, {len(chosen) - imaged} without)")
    return 0 if on_chain == len(chosen) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
