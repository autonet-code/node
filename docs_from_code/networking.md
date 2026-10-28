# Autonet peer-to-peer networking and deployment (derived from code only)

All paths are relative to `D:\autonet_codeonly`. Citations use `file:line`.

## 1. Overview

The networking layer is one libp2p host class, `AutonetHost` (`nodes/common/p2p.py:430`), running on py-libp2p 0.6.x under trio (`pyproject.toml:63-65`). Several things sit on top of it:

- **Event gossip.** World-event batches are signed and sent over GossipSub on one topic per jurisdiction (`nodes/common/event_gossip.py`).
- **Content-addressed blobs.** They are fetched over a request/response protocol and can be forwarded one hop (`p2p.py:1458-1608`, `nodes/common/blob_resolver.py`).
- **Capability advertisement.** Each node pushes a JSON capability record directly to every connected peer (`p2p.py:1288-1329`).
- **Other request/response protocols.** Inference (`/rpb/inference/1.0.0`), the robot rail, the reachability check, and older weights, activations and embeddings protocols (`p2p.py:137-154`).
- **A relay node.** It runs networking only: it is a seed, keeps the gossip mesh connected, relays blobs and answers reach checks (`nodes/relay.py`).

There is no DHT, no mDNS and no QUIC in use, even though some docstrings say so (see section 12). Peers are found only through configured or Registry-published bootstrap seeds and whatever connections already exist.

## 2. Transports

### 2.1 TCP and WebSocket

- `AutonetHost(transport=...)` accepts only `"tcp"` or `"ws"`. Anything else raises `ValueError` (`p2p.py:479-481`).
- The host listens on one address, `/ip4/<listen_host>/tcp/<listen_port>`, with `/ws` appended when the transport is ws (`p2p.py:537-540`).
- A code comment says py-libp2p runs one transport per host, picked from the listen address, so every node in a network must use the same one (`p2p.py:475-478`, `nodes/common/config.py:123-126`).
- `wss` is never served by the process itself. The intended shape is a plain `ws` listener behind an HTTPS reverse proxy that terminates TLS on 443 (`p2p.py:476-478`, `nodes/relay.py:82`). The S1 harness does exactly this with Caddy (section 10.3).
- A wss seed address looks like `/dns4/<name>/tcp/443/wss/p2p/<id>` (`scripts/s1_container_run.py:208-209`; the test fixture is at `tests/atn/test_config_registry.py:331`).
- QUIC: `enable_quic` is passed through to `new_host` (`p2p.py:553`) but defaults to False. The daemon never passes it (`atn/autonet_service.py:477-489`), so `p2p.enable_quic` in config has no effect on the daemon.

### 2.2 DNS handling

- `resolve_dns_multiaddr` rewrites a leading `/dns`, `/dns4` or `/dns6` component to `/ip4` or `/ip6`. It uses the first `getaddrinfo` result and returns the address unchanged on failure (`p2p.py:406-427`).
- `_resolve` applies that rewrite only for TCP. For ws it keeps the hostname, because TLS SNI and the HTTP Host header need it and an HTTP proxy resolves it anyway (`p2p.py:619-624`).
- Bootstrap addresses are resolved at start (`p2p.py:544`) and again on every redial (`p2p.py:662`).
- Announce addresses are resolved, and any `/p2p/` suffix is removed (`p2p.py:626-635`).

### 2.3 HTTP CONNECT proxies (`nodes/common/ws_proxy.py`)

- py-libp2p's WebSocket client reads `HTTP_PROXY` / `HTTPS_PROXY` but only speaks SOCKS5 to them (`ws_proxy.py:3-8`).
- `install()` replaces `WebsocketTransport._create_proxy_connection` once (`ws_proxy.py:54-96`):
  - For `http`/`https` proxy URLs it opens a CONNECT tunnel. The tunnel uses TLS to the proxy itself when the proxy URL is `https`, and sends Basic `Proxy-Authorization` when the URL carries credentials (`ws_proxy.py:24-51`).
  - For `wss` it then wraps the tunnel in TLS with SNI set to the target host (`ws_proxy.py:78-81`), then runs the WebSocket handshake through `trio_websocket.wrap_client_stream` (`ws_proxy.py:85-88`).
  - SOCKS URLs fall through to the library's original code (`ws_proxy.py:67-68`).
- `install()` is called only when the host's transport is `ws` (`p2p.py:482-485`).
- CONNECT is only considered successful if the status line contains ` 200` (`ws_proxy.py:48-50`), and the response header is capped at 16 KiB (`ws_proxy.py:46-47`).

### 2.4 Framing

- Every custom protocol uses a 4-byte big-endian length prefix and writes in 65000-byte chunks to stay under the Noise and Yamux limits (`p2p.py:61-130`).
- There are two framing styles:
  - Write-then-close: `_write_framed` / `_read_framed`.
  - Request/response on one stream: `_write_frame_no_close` / `_read_frame_no_close`. A comment says a full close tears down the read half on py-libp2p, so the response could never come back (`p2p.py:1640-1644`).

## 3. Host lifecycle and addresses

### 3.1 Identity

- `key_seed` (32 bytes) gives a stable ed25519 peer id through `create_new_key_pair(seed)` (`p2p.py:472-474`, `p2p.py:546-549`).
- Only the relay passes one. It reads or creates `<data-dir>/relay.key` (hex, chmod 600) (`nodes/relay.py:47-62`, `nodes/relay.py:115`).
- The ATN daemon does not pass `key_seed` (`atn/autonet_service.py:477-489`), so its libp2p peer id changes on every start (section 12).

### 3.2 Announce addresses

If any of `announce_addrs`, `auto_reach` or `enable_upnp` is set, the host replaces its own `get_addrs` method. The replacement returns the resolved announce addresses with `/p2p/<id>` appended, or falls back to the real listen addresses when there are none (`p2p.py:565-574`). Identify, the signed peer record and the capability's `listen_addrs` all read `get_addrs`, so the listen socket stays the same while peers are told a different address (`p2p.py:568-570`, `p2p.py:577-579`).

### 3.3 Background tasks started in `run()`

These run in the host's nursery (`p2p.py:598-608`):

- **`_maintain_addrs`** runs when there are bootstrap peers or announce addresses and `bootstrap_redial_interval > 0` (`p2p.py:601-603`). Every interval it:
  - re-resolves announce names and, if the result changed, re-advertises the capability (`p2p.py:650-656`);
  - redials each bootstrap peer that is not currently connected, at a freshly resolved address, after clearing that peer's cached addresses from the peerstore, with a 10 s cap per dial (`p2p.py:657-670`). A bootstrap entry with no `/p2p/<id>` part is never redialed (`p2p.py:659-661`).
- **`_auto_reach_loop`** runs only when there are no announce addresses and `auto_reach` or `enable_upnp` is set (`p2p.py:604-605`).
- The initial bootstrap dial is left to py-libp2p's `bootstrap=` argument (`p2p.py:557`). A comment says py-libp2p dials those peers only once (`p2p.py:642-645`), which is why the redial loop exists.

### 3.4 NAT handling: UPnP plus the reach protocol

The auto-reach loop (`p2p.py:692-739`) works like this:

1. **UPnP.** If enabled and the listen port is known, it uses `libp2p.discovery.upnp.upnp.UpnpManager` to discover the router and map the port as TCP. Any failure turns UPnP off for the run (`p2p.py:704-713`). py-libp2p's own UPnP is explicitly turned off because it reads `get_addrs`, which the host has replaced (`p2p.py:554-556`). The mapping is renewed every round (`p2p.py:716-718`) and removed on shutdown inside a shielded 3 s scope (`p2p.py:736-739`). Mapping needs `miniupnpc` (`pyproject.toml:65`).
2. **Reach check.** It asks up to three connected peers, with 15 s each, over `/autonet/reach/1.0.0` (`p2p.py:719-725`). If nobody answers it retries after 2 s (`p2p.py:726-728`).
3. **Decision.** If a peer reports `reachable` and the observed IP is globally routable (`_is_public_ip`, `p2p.py:396-403`), the host announces `/ip4|ip6/<ip>/tcp/<port>`. Otherwise it goes back to the listen addresses (`p2p.py:729-734`). Any change is pushed to peers at once (`p2p.py:679-690`).
4. It repeats every `reach_interval` seconds (`p2p.py:735`).

The reach handler, on the answering side (`p2p.py:754-773`):

- It reads `{"port": N}` and takes the requester's observed IP from `stream.get_remote_address()`.
- It tries a raw TCP connect to that IP and port, with a 5 s limit.
- It replies `{"ip", "reachable"}`.
- It only ever dials the requester's own observed IP (`p2p.py:151-153`), so the check cannot be used to make the node dial arbitrary hosts.

A host that nobody can reach keeps working NAT'd by dialling out (`p2p.py:699-700`).

### 3.5 GossipSub setup

GossipSub is created lazily on the first `join_guild` / `subscribe_topic` / `publish_topic` (`p2p.py:1040-1087`), with these settings:

- protocols `/meshsub/1.1.0` and `/meshsub/1.0.0`;
- degree 6, low 4, high 12;
- `heartbeat_interval=1` (a comment says 120 s used to stall mesh formation, `p2p.py:1048-1052`);
- `Pubsub(strict_signing=False)` (`p2p.py:1060-1064`).

Libp2p-level message signing is therefore off. Event batches are signed at the application layer instead (section 4.2).

Pubsub and GossipSub each run under their own `TrioManager` in the host nursery (`p2p.py:1075-1082`). Two workarounds call the private `Pubsub._handle_new_peer_safe`:

- `_reconcile_pubsub_peers` runs every 2 s and registers any connected peer that pubsub has missed (`p2p.py:807-829`);
- `_nudge_pubsub_peer` runs after every explicit connect (`p2p.py:831-848`, `p2p.py:862`).

The comments say PubsubNotifee delivery was seen to lag on ZeroTier networks.

### 3.6 Cross-thread use

- The host captures its trio token and nursery (`p2p.py:597-600`) and sets a `threading.Event` called `_ready_event` once they are ready (`p2p.py:516-517`, `p2p.py:606`).
- `publish_topic_threadsafe` queues the publish with `run_sync_soon` and never waits. This avoids a deadlock with the world lock (`p2p.py:1161-1182`).
- `subscribe_topic_threadsafe` blocks through `trio.from_thread.run` (`p2p.py:1208-1226`).
- One reader task runs per topic. It skips messages that came from this host (`p2p.py:1228-1252`).

## 4. Event gossip (`nodes/common/event_gossip.py`)

### 4.1 Topics

| Topic | Producer | Notes |
|---|---|---|
| `/autonet/world-events/<rpb_address>` | `EventGossip` (`event_gossip.py:209-211`) | `<rpb_address>` is the jurisdiction key `<chain_id>:<dao lowercased>` (`atn/config.py:746-753`, `nodes/service.py:644-654`), or `"default"` in chain-free runs (`nodes/service.py:648-649`) |
| `autonet/robot/1` | robot rail (`atn/robot/rail.py:38`, `rail.py:182-184`, `rail.py:238`) | not in the `/autonet/` namespace; the relay does not subscribe to it (section 12) |
| `/autonet/guild/<id>` | `join_guild` (`p2p.py:155`, `p2p.py:282-284`) | no reader task is ever started (section 12) |
| `/rpb/provider-registry/` | none | constant only (`p2p.py:156`) |

### 4.2 Wire format and signing

- An `EventBatch` holds schema, rpb_address, sender_pubkey, batch_seq, prev_batch_hash, timestamp and events.
- It is signed over canonical JSON (sorted keys, compact separators) (`event_gossip.py:161-172`).
- The envelope is `{"batch": <canonical body>, "signature": hex}` (`event_gossip.py:186-206`).
- `content_hash` is sha256 of the canonical payload. It chains `prev_batch_hash` and is the dedupe key (`event_gossip.py:174-177`).
- Signatures are ed25519 through `cryptography` (`event_gossip.py:62-127`). If `cryptography` is missing, the code signs with a keyed blake2b and `_verify` accepts every signature (`event_gossip.py:113-114`, `event_gossip.py:125-127`).
- The sender key is per daemon, stored at `<data_dir>/world/gossip/keypair.json` as hex in plaintext JSON. It is created on first use (`nodes/service.py:1186-1219`). It has no link to the libp2p peer id or to any on-chain identity.

### 4.3 Receive path (`event_gossip.py:514-598`)

1. Drop the message if it is on a different topic, malformed, or carries a different `rpb_address`.
2. Verify the signature.
3. If `max_skew_seconds` is set, drop batches whose signed timestamp is too far from local time. The close driver sets this from the pinned epoch rules (`nodes/common/federated_close_driver.py:930-931`).
4. Dedupe against a FIFO of 4096 hashes (`event_gossip.py:378`, `event_gossip.py:604-608`).
5. Record the highest `batch_seq` seen per sender, for diagnostics only.
6. Add the batch to the per-epoch buffer used by the federated close.
7. Skip ingesting the daemon's own batches. Otherwise pass the events to `world_service.submit_events(origin="remote")`.

### 4.4 Outbox

- `publish_events` holds local events in `_outbox` whenever the transport reports no subscribed peers, or the outbox is already non-empty (to keep ordering) (`event_gossip.py:441-464`).
- A batch is signed only when it is actually sent. A comment says this keeps its timestamp inside every receiver's skew window, and keeps unseen batches out of the local epoch set (`event_gossip.py:449-453`).
- `flush_outbox` drains the outbox oldest first while peers are present (`event_gossip.py:466-478`). It is called on every publish and on every service cycle (`nodes/service.py:323-328`).
- `has_peers` maps to `host.topic_peer_count(topic) > 0` (`event_gossip.py:285-287`, `p2p.py:885-893`).

### 4.5 Transports

- `LibP2PTransport` connects the synchronous Transport protocol to the host's threadsafe calls (`event_gossip.py:266-313`).
- `InMemoryTransport` / `InMemoryHub` are for tests (`event_gossip.py:233-344`).
- The module docstring still says the libp2p adapter is "to come" (`event_gossip.py:22-25`), but the class exists in the same file.

### 4.6 Wiring in the daemon

- `AutonetBridge.start()` starts p2p, then runs `_wire_event_gossip_when_ready` (`atn/autonet_service.py:997-1018`). That task polls for up to 30 s until both the WorldService and the host's `_ready_event` are ready, then calls `service.attach_event_gossip(host)` (`atn/autonet_service.py:1219-1252`).
- `attach_event_gossip` builds the `EventGossip` and then the federated close driver (`nodes/service.py:804-851`).
- If 30 s pass first, gossip is never attached and nothing retries (`atn/autonet_service.py:1243-1245`).

## 5. Blobs

### 5.1 Protocol `/autonet/blob/1.0.0`

- A request is `"<sha256hex>"` or `"<hash>|<hops>"` (`p2p.py:141-145`).
- `BLOB_RELAY_MAX_HOPS = 1`. A bare hash means one hop is allowed, and the server caps any requested hop count at that value (`p2p.py:1521-1525`).
- On a local miss with hops > 0, the server asks every other connected peer at the same time with `hops-1`. The timeout is `_blob_relay_timeout` = 5 s (`p2p.py:496`). Any relayed result is passed to `_blob_relay_sink` (`p2p.py:1534-1545`).
- The response is the raw bytes, or an empty frame for "not found" (`p2p.py:1547-1549`).
- The client checks that sha256 of the bytes equals the requested hash (`p2p.py:1497-1508`).
- `fetch_blob_from_peers` asks all given peers at once, and the first verified answer cancels the rest (`p2p.py:1575-1608`).

A comment says the one hop exists so that a NAT'd daemon can reach a blob held only by another NAT'd daemon, through a reachable peer that both are connected to (`p2p.py:141-144`).

### 5.2 One handler per host (`blob_resolver.py:76-158`)

- `AutonetHost.set_blob_handler` holds a single handler.
- `host_blob_server(host)` installs one `HostBlobServer` per host, stored on attribute `_autonet_blob_server`. It serves from an in-memory map first, then from registered read-only sources such as `BlobStore.get_bytes_local`, returning a source's result only when the bytes hash to the requested CID (`blob_resolver.py:129-143`).
- It also installs the relay sink so relayed blobs land in the in-memory map (`blob_resolver.py:99-102`).
- `LibP2PBlobResolver.get` checks locally first, then asks connected peers and also peers known only through capability gossip, bridging into trio with `trio.from_thread.run`. It caches verified results (`blob_resolver.py:199-262`).
- The daemon uses this resolver for chain submission (`atn/autonet_service.py:1170`) and for fetching tool blobs through `asyncio.to_thread` (`atn/autonet_service.py:1289-1298`).
- It also registers the on-disk tool and service blob stores as sources (`atn/autonet_service.py:1184-1201`).

### 5.3 `BlobStore` (`nodes/common/blob_store.py`)

- Files are stored on disk named by sha256 hex (`blob_store.py:59-68`).
- `get_bytes` checks disk first, then each `peer_urls` entry over HTTP as `GET <url>/blob/<hash>` with hash verification (`blob_store.py:90-124`).
- `get_bytes_async` adds a P2P fallback after that (`blob_store.py:147-179`).
- An optional HTTP server, `start_server(port=9100)`, serves `/blob/<hash>` on 0.0.0.0 (`blob_store.py:253-291`). No caller in the code ever starts it (section 12).

## 6. Capability advertisement and discovery

- `NodeCapability` is JSON with sorted keys. It carries peer_id, node_id, roles, gpu fields, modules_hosted, listen_addrs, agents, model_state, inference and timestamp (`p2p.py:239-272`).
- `advertise_capability` opens a `/autonet/capability/1.0.0` stream to every connected peer and writes the record (`p2p.py:1288-1315`).
- How often it runs:
  - the daemon, every `p2p.capability_advertise_interval` seconds (default 60) (`atn/autonet_service.py:472`, `atn/autonet_service.py:509-512`);
  - the relay, every `--advertise-interval` seconds (`nodes/relay.py:136-139`);
  - also on any address change (`p2p.py:656`, `p2p.py:690`).
- On receipt the record is stored under the `peer_id` field inside the payload, not under the stream's authenticated remote peer (`p2p.py:1317-1329`). Records are never expired.
- The capability's `listen_addrs` are never used to dial anyone. Discovery beyond the seeds depends on connections that already exist.
- `discover_inference_providers` filters by price and model and sorts known-latency peers first (`p2p.py:1877-1970`). `RPBNetworkProvider.discover_providers` does its own scan of `_known_capabilities`, preferring sponsors (`atn/providers/rpb.py:81-146`).
- Latency: `PeerLatencyTracker` exists (`p2p.py:303-388`), but nothing in non-test code ever calls `ping_peer` / `ping_all`. Every latency is therefore unmeasured: `None` in `discover_inference_providers`, and `9999` in the RPB provider (`rpb.py:124-128`).

## 7. Request/response protocols

| Protocol | Handler | Used by |
|---|---|---|
| `/autonet/weights/1.0.0` | `p2p.py:933` (sha256-prefixed payload) | no non-test caller |
| `/autonet/activations/1.0.0` | `p2p.py:1009` | no non-test caller |
| `/autonet/embeddings/1.0.0` | `p2p.py:2010` | `nodes/common/distributed_jepa.py:170,496,525` |
| `/autonet/capability/1.0.0` | `p2p.py:1317` | section 6 |
| `/autonet/blob/1.0.0` | `p2p.py:1517` | section 5 |
| `/rpb/inference/1.0.0` | `p2p.py:1668` (`_inference_handler`, `{"error"}` if unset) | sponsor handler set directly as `host._inference_handler` (`atn/autonet_service.py:496-499`); client `atn/providers/rpb.py:198` |
| `/autonet/robot/1.0.0` | `p2p.py:1760` | robot rail (`atn/robot/rail.py:160-174`, `rail.py:237`) |
| `/autonet/reach/1.0.0` | `p2p.py:754` | section 3.4 |

The robot rail finds a target agent's peer id through the on-chain `PeerAttribution` that is stored on the host as `_peer_attribution` (`atn/robot/rail.py:153-158`, `atn/autonet_service.py:1148-1150`).

## 8. Seeds: where bootstrap peers come from

The order, highest priority first, as applied in `start_p2p` (`atn/autonet_service.py:464-471`):

1. **`p2p.bootstrap_peers` from the nodes-side config.** That config is `autonet.config_path` if set, else `AUTONET_CONFIG`, else `./autonet.yaml`, else `~/.autonet/config.yaml` (`nodes/common/config.py:381-427`, `atn/config.py:179`). `AUTONET_P2P_BOOTSTRAP` (a comma list) overrides it (`nodes/common/config.py:318`).
2. **The jurisdiction's Registry seeds** (`RPBConfig.bootstrap_peers`). These come from the key `jurisdiction.<id>.bootstrap`, a comma-separated list of multiaddrs (`atn/config.py:628-632`, `atn/config.py:270-272`). The registry document is resolved at join time, never at boot (`atn/config.py:939-998`):
   - `ATN_REGISTRY_URL` (HTTP fetch, 4 s timeout);
   - else `ATN_JURISDICTION_REGISTRY` plus `ATN_JURISDICTION_RPC`, or the saved `<data_dir>/jurisdiction.json` selection, read on chain through the DAO Registry's `getAllKeys` / `getAllValues` (`atn/config.py:889-936`);
   - else the packaged `atn/registry.json`.
   
   A Registry list can replace an empty `bootstrap_peers` list (`atn/config.py:1040-1043`). `start()` resolves the registry (`ensure_network_config`) before it calls `start_p2p` (`atn/autonet_service.py:912`, `atn/autonet_service.py:997`).
3. **`COMMUNITY_BOOTSTRAP_PEERS`.** This list is currently empty (`nodes/common/peer_attribution.py:56-60`, `peer_attribution.py:205-215`).

Neither the repo-root `registry.json` nor `atn/registry.json` contains a `bootstrap` key (a grep found none). Seeds therefore come only from config, env, or an on-chain or URL registry. The S0 and S1 harnesses put the relay's address into the Registry at deploy time through `BOOTSTRAP=<seed>` passed to the external `deploy-autonet-jurisdiction.js` (`scripts/s0_local_run.py:219-247`).

## 9. Relay node (`nodes/relay.py`)

- **Start command:** `atn relay ...` (`atn/cli.py:995-1000`) or `python -m nodes.relay`.
- **Flags and their env twins** (`relay.py:65-96`):
  - `--port`: `AUTONET_P2P_PORT`, default 4001
  - `--host`: `AUTONET_P2P_HOST`, default 0.0.0.0
  - `--announce`: `AUTONET_P2P_ANNOUNCE`
  - `--bootstrap`: `AUTONET_P2P_BOOTSTRAP`
  - `--jurisdiction` (repeatable): `AUTONET_RELAY_JURISDICTIONS` (comma list)
  - `--data-dir`: `AUTONET_RELAY_DIR`, default `~/.autonet-relay`
  - `--transport tcp|ws`: `AUTONET_P2P_TRANSPORT`
  - `--no-upnp` and `--advertise-interval` (default 60) have no env twin.
- **Default jurisdiction:** with no jurisdiction given, it uses `atn.config.boot_jurisdiction_key()`. That is `ATN_JURISDICTION_REGISTRY` if set, else the saved selection, else the packaged registry's `autonet` entry (`relay.py:88-95`, `atn/config.py:829-886`).
- **Host settings:** stable key from `relay.key`, UPnP on unless `--no-upnp`, `auto_reach` only when there is no `--announce`, roles `["relay"]` (`relay.py:107-118`).
- **Blobs:** it serves disk blobs from `<data>/blobs` and replaces the relay sink so relayed blobs are written to disk (`relay.py:120-125`).
- **Gossip:** it subscribes with a no-op handler to `topic_for_rpb(key)` for each jurisdiction, so that the GossipSub mesh forwards between peers that are not connected to each other (`relay.py:127-132`, docstring `relay.py:6-13`).
- **Output:** it prints `relay address: <multiaddr>` for each address, and the harnesses scrape that line (`relay.py:134-135`, `scripts/s0_local_run.py:791-794`).

## 10. Deployment shapes

### 10.1 Desktop / interactive daemon

- `atn` runs `run_cli()`, an interactive console, when stdin is a TTY (`atn/cli.py:780-795`).
- Autonet, and with it p2p, starts only when `autonet.enabled: true` is set, or when an agent is or becomes registered on chain (`atn/runtime/__init__.py:840-855`, `__init__.py:874-901`).
- Agents register with the daemon's current libp2p peer id (`atn/ws_server.py:5737-5804`).
- Other listeners, which are not part of the P2P layer:
  - the local owner WS (default `ws://127.0.0.1:7700`, `atn/cli.py:952`);
  - an optional remote wallet-auth WS (`ATN_REMOTE_WS_HOST`/`_PORT`, `atn/cli.py:585-610`);
  - an integration listener (`ATN_INTEGRATION_WS`, `_HOST`, `_PORT`, default 7710, `atn/cli.py:564-582`);
  - optional publication of `public_ws_endpoint` to a Firestore agent directory (`atn/runtime/__init__.py:865-877`).

### 10.2 Headless daemon

- `atn --headless`, or any run with a non-TTY stdin, waits for the shutdown event instead of reading input (`atn/cli.py:780-793`).
- `python -m nodes` runs `nodes.service.AutonetService` directly (`nodes/__main__.py:41-45`). It never constructs an `AutonetHost` (no `AutonetHost(` call in `nodes/service.py`), so that entry point has no P2P.

### 10.3 Docker (`docker/Dockerfile.daemon`)

- Two-stage build on python:3.11-slim. State lives on one volume at `/data/atn`, with `~/.atn` symlinked to it.
- The entrypoint is tini plus `atn.guest_launcher`, and the default command is `atn --headless` (`Dockerfile.daemon:111-114`).
- `AUTONET_P2P_PORT=4001` and `EXPOSE 7710 4001` (`Dockerfile.daemon:105-109`). The comments name 4001 as the only public port (`Dockerfile.daemon:15-21`, compose snippet `:37-62`).
- No transport, announce or bootstrap env is set. A container behind Docker NAT therefore relies on UPnP or auto-reach (which usually fails inside a bridge network), or on `AUTONET_P2P_ANNOUNCE` set by the operator.
- The build copies `README.md` (`Dockerfile.daemon:74`). That file does not exist in this docs-stripped copy, so building from this copy would fail at that step.

### 10.4 Relay node

See section 9. The intended layout is a small public server with a stable `--announce /dns4/<host>/tcp/<port>[...]` (`relay.py:17`), or a ws listener behind an HTTPS proxy on 443 (`relay.py:82`).

### 10.5 Harnesses and the topology they assume

**S0 (`scripts/s0_local_run.py`)** runs on one machine:

- a hardhat chain on 8545 with interval mining;
- one relay on `127.0.0.1:4101` using ws, with `--no-upnp`;
- four `python -m atn --headless` daemons, each with a redirected HOME, WS port `7811+10i`, P2P port `4111+i`, `AUTONET_P2P_HOST=127.0.0.1`, `AUTONET_P2P_TRANSPORT=ws` (`s0_local_run.py:63-69`, `:286-300`).

Proxy, registry and bootstrap env are cleared first (`s0_local_run.py:134-139`). Daemons find the relay only through the Registry key `jurisdiction.autonet.bootstrap` = `/dns4/localhost/tcp/4101/ws/p2p/<id>` (`s0_local_run.py:214-216`). The run order matters:

1. A "relay-probe" run with a dummy jurisdiction fixes the relay's peer id in `relay.key` (`s0_local_run.py:782-798`).
2. The jurisdiction is deployed with that seed.
3. The relay restarts with the same data dir and the real key `31337:<governor>` (`s0_local_run.py:803-805`).
4. The daemons get `ATN_JURISDICTION_REGISTRY=<registry>` (`s0_local_run.py:293-294`).

The topology is a star around the relay. Daemons never dial each other directly, because no daemon announces a public address.

**S1 (`scripts/s1_container_run.py`)** is S0 with every daemon in a container behind HTTPS-only egress (`s1_container_run.py:1-31`):

- `s1-caddy` stands in for Cloudflare. It terminates TLS for `relay.autonet.test` and reverse-proxies to the host relay on 4101. It also serves `rpc.autonet.test`, but only under a random token path, and answers 403 to anything else (`s1_container_run.py:82-101`).
- `s1-proxy` runs `scripts/s1/gateway.py proxy`, an allowlisting CONNECT proxy that permits only `relay.autonet.test:443,rpc.autonet.test:443` and logs `ALLOW`, `DENY` or `FAIL`. Non-CONNECT requests get 403 (`gateway.py:48-89`, `s1_container_run.py:120-124`).
- `s1-ctl` runs `gateway.py forward` (plain TCP splicing) so the harness can reach each daemon's owner WS (`s1_container_run.py:125-133`, `gateway.py:92-110`).
- The daemons sit on an `--internal` network with these env values: `AUTONET_P2P_TRANSPORT=ws`; `HTTP(S)_PROXY=http://s1-proxy:3128`, also in lowercase; `NO_PROXY=localhost,127.0.0.1`; `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` pointing at Caddy's local CA; RPC through `https://rpc.autonet.test/<token>` (`s1_container_run.py:175-205`). Each container runs its own forward from 7800 to 127.0.0.1:7700, because the owner listener binds loopback (`s1_container_run.py:191-197`).
- The seed is `/dns4/relay.autonet.test/tcp/443/wss/p2p/<id>`, "the shape the real run uses" (`s1_container_run.py:21-22`, `:208-209`).

This shows the intended WAN topology: daemons on cloud VMs that can only reach the internet through an HTTPS agent proxy, dialling out over wss to one relay behind Cloudflare. The relay carries all gossip and blob traffic.

**Older smoke tests:**

- `scripts/cross_machine_smoke.py` uses env `SUBSTRATE_LISTEN_PORT`, `SUBSTRATE_LISTEN_HOST`, `SUBSTRATE_RPB`, `SUBSTRATE_DATA`, `SUBSTRATE_BOOTSTRAP`, `SUBSTRATE_EMBEDDING_DIM`, `SUBSTRATE_OBS_INTERVAL`, `SUBSTRATE_CLOSE_INTERVAL`, `SUBSTRATE_DURATION` and `SUBSTRATE_OBS_AXIS` (`cross_machine_smoke.py:54-62`, `:153-156`). It runs over TCP on ZeroTier IPs (example `/ip4/10.241.68.122/tcp/4001`).
- `scripts/_ec2_launch_smoke2.sh` launches it on EC2 with a bootstrap address as `$1`.
- `scripts/cross_machine_inference_smoke.py` uses `INF_BOOTSTRAP` (`:31`, `:203`).
- `scripts/e2e_live/harness.py` writes an `autonet.yaml` with `listen_host: 127.0.0.1` and no bootstrap peers, and points `AUTONET_CONFIG` at it (`harness.py:405-431`).

## 11. Environment variables and config keys read by the networking layer

### 11.1 Nodes config (`nodes/common/config.py`)

| Key | Default | Env override | Read by daemon `start_p2p`? |
|---|---|---|---|
| `p2p.listen_port` | 0 (random) | `AUTONET_P2P_PORT` | yes (`autonet_service.py:462`) |
| `p2p.listen_host` | 0.0.0.0 | `AUTONET_P2P_HOST` | yes (`:463`) |
| `p2p.bootstrap_peers` | [] | `AUTONET_P2P_BOOTSTRAP` (comma list) | yes (`:469-471`) |
| `p2p.announce_addrs` | [] | `AUTONET_P2P_ANNOUNCE` (comma list) | yes (`:483`) |
| `p2p.bootstrap_redial_interval` | 30 (0 = dial once) | none | yes (`:484`) |
| `p2p.transport` | tcp | `AUTONET_P2P_TRANSPORT` | yes (`:488`) |
| `p2p.enable_upnp` | True | none | yes (`:485`) |
| `p2p.auto_reach` | True | none | yes (`:486`) |
| `p2p.reach_interval` | 300 | none | yes (`:487`) |
| `p2p.capability_advertise_interval` | 60 | none | yes (`:472`) |
| `p2p.enable_quic` | False | none | no |
| `p2p.ping_interval` | 30 | none | no (never read) |
| `blob_store.data_dir` | `~/.autonet/blobs` | `AUTONET_BLOB_DIR` | `nodes/service.py:346` |
| `blob_store.peer_urls` | [] | none | `nodes/service.py:350-354` |
| `blob_store.server_port` | 9100 | none | never read |
| `jurisdiction_key` | "" | set by the bridge (`autonet_service.py:979`) | names the gossip topic |

The source table is `nodes/common/config.py:113-134` and `:299-328`. The config file is located through `AUTONET_CONFIG` (`config.py:399`). `AUTONET_DATA_DIR` maps to `data_dir`.

### 11.2 Other variables

- **Relay only:** `AUTONET_RELAY_JURISDICTIONS`, `AUTONET_RELAY_DIR`. It also reads the `AUTONET_P2P_*` twins directly with `os.environ` (`relay.py:66-87`).
- **Seed resolution:** `ATN_REGISTRY_URL`, `ATN_JURISDICTION_REGISTRY`, `ATN_JURISDICTION_RPC` (`atn/config.py:575-577`, `:861-904`). Registry key `jurisdiction.<id>.bootstrap`.
- **Proxy:** `HTTP_PROXY`, `HTTPS_PROXY` and lowercase variants, read by py-libp2p's ws transport (`ws_proxy.py:3-4`). S1 also sets `NO_PROXY`, `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE`. The ws TLS path uses `ssl.create_default_context()` (`ws_proxy.py:30`, `:79`), which honours `SSL_CERT_FILE`.
- **Docker image:** `AUTONET_P2P_PORT=4001`.

## 12. Inconsistencies and dead code

1. **The daemon's peer id is not persistent.** `start_p2p` never passes `key_seed` (`atn/autonet_service.py:477-489`), so each restart gets a new libp2p id. Agents register with the current one (`atn/ws_server.py:5737-5804`). `OnChainService.update_peer_id` exists (`atn/on_chain.py:1039-1056`) but nothing calls it. After a restart, on-chain `getAgentPeerId` is stale, so robot-rail routing (`atn/robot/rail.py:153-158`) and `PeerAttribution` point at a dead id. Only the relay persists its key.
2. **Auto-reach announces a TCP address even on ws.** `_auto_reach_loop` announces `/ip4/<ip>/tcp/<port>` with no `/ws` suffix (`p2p.py:731-732`), even when `transport == "ws"`. On a ws network that address is undialable. The reach check itself is a raw TCP connect (`p2p.py:766`), so it cannot tell whether a ws upgrade would work.
3. **Docstrings describe features that do not exist.** The `p2p.py` docstrings promise Kademlia DHT discovery, "QUIC transport (preferred)", DHT capability advertisement, and a constructor `AutonetHost(node_id=..., config=p2p_config)` (`p2p.py:4-19`, `:435`). None of that exists: there is no `config` parameter, no DHT, and QUIC is off by default. `peer_attribution.py:33-34` says the DHT is unused. `blob_resolver.py:177` mentions mDNS, and there is no mDNS code. `peer_attribution.py:14-15` says new peers learn about everyone through libp2p's natural peer exchange, but nothing dials gossiped `listen_addrs`.
4. **`_P2P_AVAILABLE` does nothing.** `TProtocol(...)` is called at module level (`p2p.py:137-150`), so a missing libp2p raises `NameError` on import anyway.
5. **Guild gossip is dead.** `join_guild` subscribes but never starts a reader, so `_guild_handlers` is never called (`p2p.py:1089-1113`). `publish_guild_message`, `leave_guild` and `get_guild_for_module` have no non-test callers. `PROVIDER_REGISTRY_TOPIC` is unused (`p2p.py:156`).
6. **Training-era protocols are unused.** The weights and activations protocols have no callers outside `p2p.py`. Embeddings are used only by `distributed_jepa.py`. `nodes/service.py:308-310` says JEPA is retired.
7. **Latency is never measured.** `PeerLatencyTracker.ping_*` is never called, and `p2p.ping_interval` is never read. Every "latency-aware" sort (`p2p.py:1964-1969`, `atn/providers/rpb.py:139-140`) falls back to price, or to sponsor flag then 9999.
8. **`enable_quic` is a no-op for the daemon.** It is a config key (`config.py:127`, `autonet.yaml:63`) but is not passed to the host.
9. **Blob-store leftovers.** `blob_store.server_port` / `BlobStore.start_server` are never used, so the HTTP blob server never runs, while `peer_urls` clients still exist. `BlobStore._serve_blob_for_p2p` (`blob_store.py:143-145`) and `announce_blobs_to_peers` (`blob_store.py:201-223`) have no callers; the latter would also append `blob:<hash>` entries to `modules_hosted` every time it ran. `InferenceP2PConfig` (`config.py:137-144`) is not part of `AutonetConfig` and is never read.
10. **Capability records are unauthenticated.** A record is stored under the `peer_id` claimed inside its JSON, not the stream's remote peer (`p2p.py:1321-1322`), so any peer can overwrite another's entry. Records never expire. `from_bytes` uses `cls(**d)` (`p2p.py:270-272`), so a peer sending an unknown top-level field has its record dropped, despite the "wire format can evolve" comment (`p2p.py:253-255`).
11. **Gossip trust rests only on application signatures.** Pubsub uses `strict_signing=False` (`p2p.py:1063`). The ed25519 key is self-asserted and tied to neither the peer id nor the chain. If `cryptography` is missing, `_verify` returns True for everything (`event_gossip.py:125-127`). The keypair is stored in plaintext JSON (`nodes/service.py:1212-1218`).
12. **The relay does not forward robot-rail gossip.** The relay subscribes only to world-event topics (`relay.py:127-128`), so the GossipSub mesh will not carry `autonet/robot/1` (`atn/robot/rail.py:38`) between NAT'd daemons. The topic also breaks the `/autonet/...` naming used everywhere else.
13. **Relay docstring is incomplete.** `relay.py:24-26` says "every flag has an env twin", but `--no-upnp` and `--advertise-interval` have none, and the twin `AUTONET_P2P_TRANSPORT` is not in that list.
14. **Peer attribution is refreshed once.** It refreshes once at startup (`atn/autonet_service.py:1139-1140`). `maybe_refresh` is never called, so agents registered later are unknown to the robot rail until a restart. `is_registered_peer` and `agent_for_peer` are unused, so the "ignore unregistered peers for consensus" idea (`peer_attribution.py:17-18`) is not enforced anywhere.
15. **Gossip attaches once or never.** `_wire_event_gossip_when_ready` (and the peer-attribution and chain-submission waiters) give up silently after 30 s with no retry (`atn/autonet_service.py:1228-1245`, `:1157-1168`, `:1110-1119`).
16. **The inference client looks unbridged.** `RPBNetworkProvider.send` awaits `self._p2p.request_inference(...)` directly (`atn/providers/rpb.py:198`). That is a trio coroutine (it uses `self._host.new_stream`), called from the asyncio provider path. Every other cross-runtime call bridges through `_trio_token` (`atn/robot/rail.py:166-173`, `blob_resolver.py:248-259`). This looks broken unless a caller bridges it elsewhere. Not verified at runtime.
17. **The sponsor handler bypasses its setter.** It is installed by assigning the private `host._inference_handler` (`atn/autonet_service.py:497`, `:783`) instead of calling `set_inference_handler` (`p2p.py:1704-1716`).
18. **Redial can miss bootstrap entries.** Entries without `/p2p/<id>` are dialled once by py-libp2p and never again (`p2p.py:659-661`).
19. **`autonet.yaml` uses defaults for several keys.** It sets `enable_upnp` and `auto_reach` but omits `transport`, `announce_addrs`, `reach_interval` and `bootstrap_redial_interval` (`autonet.yaml:59-67`). A comment in `config.py:202-204` notes that a daemon reads `autonet.yaml` only from its cwd, so installed daemons usually run on dataclass defaults plus env.
20. **Docker image defaults do little in a bridge network.** The image ships with UPnP and auto-reach on and TCP transport, but no announce address. In a Docker bridge network neither UPnP nor auto-reach can usually give a public address, while the S1 harness overrides the transport to ws.
21. **Stale docstring in `event_gossip.py`.** It says the libp2p transport is future work (`event_gossip.py:22-25`, `:237`), but `LibP2PTransport` is defined in the same file and is used in production (`nodes/service.py:834`).
22. **Unused methods.** `connect_to_info` (`p2p.py:868`) and `fetch_blob_from_any_peer` have no daemon-path callers. The latter is used by `BlobStore.get_bytes_async` and `sharding.py:704`, which belong to the JEPA/sharding path.
