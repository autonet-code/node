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
]


def _spec(name: str, description: str, params: list[str], ask: str,
          author: str) -> dict:
    return {
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
        "ask": {
            "token": "0x4C4dAEd19B98ddaEc6cC421b6a781Bd3fBB7af00",
            "amount": ask,
            "unit": "per_item",
        },
    }


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
        print(f"registered {ok}/{len(SERVICES)} services "
              f"across {len({s[4] for s in SERVICES})} providers")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
