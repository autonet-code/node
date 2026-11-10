# P2P networking: the mesh

Most daemons run on home or cloud machines that cannot accept inbound
connections. This page covers how they still find each other, connect
directly where they can, gossip, fetch blobs and open service streams to
each other. Code: `nodes/common/p2p.py` (AutonetHost),
`nodes/common/p2p_mesh.py`, `nodes/common/ws_proxy.py`, `nodes/relay.py`,
`nodes/common/blob_resolver.py`, `nodes/common/service_stream.py`,
`nodes/common/config.py`. Library: py-libp2p 0.8.1 (pinned).

## The shape

- Every host listens on **TCP and WebSocket at once** and dials whichever
  address a peer offers, TCP first for direct links (hole punching is TCP
  only), ws/wss for relays behind an HTTPS proxy and for proxied machines.
- **Relays** (`atn relay`) run circuit relay v2 as hop, with reservations.
  A NAT'd daemon reserves a slot on up to two relays it is connected to and
  advertises a `/p2p-circuit` address through each, so any peer can reach
  it through a relay.
- **DCUtR** then upgrades a relayed connection between two NAT'd daemons to
  a direct TCP one by hole punching; the relayed leg closes after a minute.
- **A Kademlia DHT** runs on every daemon and relay. It finds peers (a
  lookup for a random key each few mesh rounds names peers we did not
  know) and blob providers (below). A peer we fail to dial leaves the
  routing table, so a dead relay does not stall lookups.
- **Gossip** (GossipSub) rides the same links. A pubsub stream that ran on
  a connection that closed while the peer stays connected another way (a
  relayed leg DCUtR replaced, or whose relay went down) is reopened on a
  live connection; py-libp2p kept the dead stream and gossip to that peer
  stopped.
- **The mesh loop** (every 15 s) keeps the relay reservations, feeds every
  new connection to the DHT, remembers connected peers in the peer cache,
  and dials towards `target_peers` (8) from capability gossip, the DHT
  routing table, the peerstore and the peer cache. Failed peers back off
  (30 s doubling to 30 min).
- **The connection manager** prunes unprotected connections above
  `max_peers` (48; relays default to 2000). Seeds and the relays we hold
  reservations on are protected.

In the real-network test every daemon was NAT'd and connected only to the
relays: a star. Now the relays are where daemons meet; most links between
daemons end up direct, and the network keeps gossiping and serving blobs
with every relay down.

## Config

| Field (`p2p.*`) | Env | Meaning |
|---|---|---|
| `listen_port` | `AUTONET_P2P_PORT` | WebSocket port (0 = random; a daemon keeps one in `<data>/p2p.port`) |
| `tcp_port` | `AUTONET_P2P_TCP_PORT` | TCP port (0 = `listen_port` + 1; a daemon keeps one in `<data>/p2p.tcp_port`) |
| `listen_host` | `AUTONET_P2P_HOST` | Bind address |
| `transport` | `AUTONET_P2P_TRANSPORT` | `tcp+ws` (default), `ws` or `tcp` |
| `bootstrap_peers` | `AUTONET_P2P_BOOTSTRAP` | Comma list of multiaddrs (`.../p2p/<peer id>`) |
| `announce_addrs` | `AUTONET_P2P_ANNOUNCE` | Comma list of addresses peers should dial us on |
| `bootstrap_redial_interval` | | Seconds between redials of unconnected bootstrap peers (0 = dial once) |
| `seed_refresh_interval` | | Seconds between re-reads of the Registry seed list on a running daemon (default 300, 0 = read at join only) |
| `enable_upnp` | | Map both listen ports on the home router (default on) |
| `auto_reach` | | Find and announce our public addresses when no `announce_addrs` is set (default on) |
| `reach_interval` | | Seconds between reachability checks (default 300) |
| `target_peers` | | Connections the mesh loop dials towards (default 8) |
| `max_peers` | | Connections above which unprotected ones are pruned (default 48) |
| `dht` | | Run the DHT (default on) |

The peer cache is `<data>/p2p_peers.json` (relays: `relay_peers.json`).

## Reachable and NAT'd daemons

Home users need no network setup. With no `announce_addrs`:

1. **UPnP:** the daemon asks the home router to forward both listen ports
   and renews the mappings every check.
2. **Reach check** (`/autonet/reach/1.0.0`): the daemon asks a peer it is
   directly connected to what IP it sees it on and whether a fresh TCP
   connect to that IP and each listen port works. The peer dials only the
   requester's observed IP. For each port that works on a public IP the
   daemon announces `/ip4/<ip>/tcp/<tcp port>` and
   `/ip4/<ip>/tcp/<ws port>/ws`; if a later check fails, it goes back to its
   listen addresses. Repeating the check every `reach_interval` follows a
   changed home IP with no DNS.
3. Otherwise the daemon is NAT'd: it advertises its LAN addresses (so a
   peer on the same LAN connects directly) and its circuit addresses, and
   DCUtR makes its links direct where both NATs allow it.

Whether a daemon announces a dialable address is advertised in its
capability (`NodeCapability.reachable`: configured announce addresses, or
one the reach check confirmed). Buyers dial a reachable provider directly;
a NAT'd one is reached through a relay. The app shows it to the owner
(`list_services` returns `reachability`, with advice when the daemon is not
reachable), and catalog rows show it per provider (`reachable`).

- A **reachable** daemon (public IP, port forward or tunnel) sets
  `announce_addrs` to its public addresses. They replace the listen
  addresses in identify, the signed peer record, DHT provider records and
  the gossiped capability; the listen sockets are unchanged.
- **Advertised addresses** never include observed ones (behind NAT those
  name the port of an outbound socket). `0.0.0.0` listen addresses are
  expanded to the machine's interface addresses.
- **DHT mode:** server when peers can reach us (a relay, an announced or
  confirmed public address, a live relay reservation, or a direct link to
  another daemon), client otherwise; peers learn the switch through an
  identify push.
- `/dns4`, `/dns6` and `/dns` names in TCP addresses are resolved to `/ip4`
  or `/ip6` before dialling and again every `bootstrap_redial_interval`, so
  a reachable daemon should announce a name (dynamic DNS or a tunnel name).
  WebSocket addresses keep their names (see below).
- **IP changes.** A NAT'd daemon whose IP changes loses its connections and
  redials its seeds and cached peers; nobody needs its new address. A
  reachable daemon whose announce name now resolves elsewhere advertises
  the new address at once (identify push), and peers that bootstrap to it
  redial the name's current address.

## Dialling

py-libp2p's swarm skips `/p2p-circuit` addresses, so the host replaces the
swarm's `dial_peer`: a peer with circuit addresses is dialled on its direct
addresses first and through its relays 2 s later (or as soon as the direct
dial fails); the first connection wins, and concurrent dials to one peer
share one attempt. Every dial (connect, a stream to a peer we are not
connected to, the DHT, gossip) goes through it. A fetch or service call to a
peer we are not connected to first dials it at the addresses we know
(capability gossip, the peerstore) or the ones the DHT finds for it.

## Seeds, the peer cache and relay nodes

- **Seeds come from the DAO Registry.** `jurisdiction.<id>.bootstrap` holds
  comma-separated multiaddrs, changed by proposal like the charter (werule
  validates them as `/ws` or `/wss`). A daemon uses them when
  `p2p.bootstrap_peers` (or `AUTONET_P2P_BOOTSTRAP`) is not set locally;
  local config always wins. A running daemon re-reads the key every
  `seed_refresh_interval` seconds and dials seeds it is not connected to.
  A failed read or an emptied key keeps the seeds it has.
- **Rejoining.** While no peer is connected the daemon redials its seeds and
  the newest cached peers at once, then with backoff (5 s doubling to
  2 min). The cache keeps circuit addresses too, so a restart with every
  seed down rejoins through any cached peer still up.
- **Relay nodes** (`atn relay`, a daemon mode) are the natural seeds:
  networking only, light enough for a small server. A relay keeps a stable
  peer id (`<data>/relay.key`), serves circuit relay v2 (reservations last
  an hour and are refreshed; a relayed connection is unlimited unless the
  operator caps it, below), serves the DHT, subscribes to each jurisdiction it serves
  (`--jurisdiction <chain_id>:<DAO>`) so gossip crosses it, keeps the state
  blobs daemons offer it after each close (anchor payload, carry-over,
  agent mints, over `/autonet/blob-offer/1.0.0`; it fetches what it lacks
  from the offerer) and provides them in the DHT, so a joiner can install
  the anchored state when every daemon that computed it is offline (issue
  #44), forwards service calls one hop, and answers reachability checks.
  No close, no chain, no agents, no model. It prints its full multiaddrs
  on start.

```
atn relay --port 4001 --tcp-port 4002 \
    --announce /dns4/relay.example.org/tcp/443/wss,/ip4/203.0.113.7/tcp/4002
```

`--port` is the WebSocket port, `--tcp-port` the TCP one (default
`--port` + 1). Without `--jurisdiction` it relays the jurisdiction this
install would join. To make a relay a seed, the DAO writes its wss address
into `jurisdiction.<id>.bootstrap` by proposal.

Relayed connections are unlimited by default (Decision 2026-11-10: a relay
gives bandwidth the way a torrent seed does, and relays are not paid). Its
operator may cap them; the caps reach the host as `AutonetHost.relay_limits`:

| Flag | Env | Meaning (0 = the default) |
|---|---|---|
| `--circuit-max-duration` | `AUTONET_RELAY_CIRCUIT_MAX_DURATION` | Seconds one relayed connection may stay open (default: no limit) |
| `--circuit-max-bytes` | `AUTONET_RELAY_CIRCUIT_MAX_BYTES` | Bytes one relayed connection may carry, both ways together (default: no limit) |
| `--max-circuits` | `AUTONET_RELAY_MAX_CIRCUITS` | Relayed connections open at once to one daemon (default 64) |
| `--no-circuits` | `AUTONET_RELAY_NO_CIRCUITS` | Hold no reservations and relay no connections (gossip, blobs and service requests are still relayed) |

Circuit relay v2 stacks default to tight limits (go-libp2p: 2 minutes and
128 KiB per relayed connection), which would end any stream within minutes;
none apply here. "No limit" on bytes is a count no circuit reaches
(`RELAY_UNLIMITED_BYTES`), not 0: py-libp2p 0.8.1 reads a data limit of 0
as no shared budget, but its per-reservation fallback then allows 0 bytes.

## WebSocket, HTTPS proxies and proxied machines

- **Hiding a relay's address.** A relay can sit behind an HTTPS proxy such
  as Cloudflare: it serves plain WebSocket on the origin, the proxy
  terminates `wss` on 443, and the published seed is
  `/dns4/<relay hostname>/tcp/443/wss/p2p/<peer id>`. Firewall the origin to
  the proxy's ranges.
- **Hole punching needs real addresses.** Behind Cloudflare the relay sees
  the proxy's IPs, so it cannot tell a daemon its public address. DCUtR
  needs at least one directly connected peer that sees the daemon's real
  IP: expose the relay's TCP port directly as well (and announce it), or
  run any reachable daemon.
- **HTTPS-only machines.** A machine with `HTTPS_PROXY`, `HTTP_PROXY` or
  `ALL_PROXY` set runs WebSocket only and dials every peer through the
  proxy: HTTP CONNECT (`http://`, `https://`) or SOCKS5 (`socks5://`,
  `socks5h://`), both in `nodes/common/ws_proxy.py`; `NO_PROXY` is
  honoured. The proxy resolves names, so the dialer needs no DNS. Such a
  daemon is reachable through its relay reservations only (no hole
  punching over WebSocket).
- **Names stay names.** `/dns4` WebSocket addresses are not resolved
  locally: TLS (SNI) and the HTTP Host header carry the hostname, which the
  proxy in front of a relay routes on.
- **Certificates are verified** on `wss` dials (system trust store, or
  `SSL_CERT_FILE`). Peer authentication is still libp2p's: the connection
  is authenticated by the peer id (Noise), whoever terminates TLS.
- Each WebSocket connection runs in its own task group, so a rejected
  handshake (a proxy answering 502 for a dead relay) fails that dial alone.

## Blobs

- **Providers.** A daemon announces in the DHT (`provide`) every blob a
  tool needs that it holds: a published tool's manifest and code before
  its registration event is gossiped (`ToolStore.blob_announcer`, bounded
  at 15 s, the registration goes out regardless), an adopted tool's once
  installed, and every tool blob it holds at start. It also provides the
  evidence of its own malware flags (`ToolStore.store_evidence`, when the
  flag is written and again at start), so any daemon can fetch a flag's
  evidence by its hash while the reviewer's daemon is online; malware
  evidence only (other review blobs are not announced), and no copy is
  offered to relays. Relays provide the blobs they keep. Records are re-announced every 30 min and whenever our
  addresses change (a new reservation), so they carry dialable addresses.
- **Fetch** (`AutonetHost.fetch_blob_from_network`, behind
  `LibP2PBlobResolver` and `BlobStore`): ask the connected peers and look
  the blob's providers up in the DHT at the same time, and ask each
  provider found (dialled directly or through a relay circuit). The first
  answer whose sha256 matches wins.
- **No forwarding.** `/autonet/blob/1.0.0` answers from what a peer holds;
  a miss is an empty frame. (The one-hop forward is gone: a `<hash>|<hops>`
  request from an older daemon is answered as `<hash>`.)
- The daemon's on-disk tool and service blob stores are served as sources
  on the host's single blob handler, so manifests and code a daemon
  published stay fetchable after it restarts.
- **The search model** (`atn/search_model.py`). Its two pinned files are
  blobs like any other: a daemon holding them verified serves them from
  disk and provides their sha256 (`SearchModel.serve_on`, again once the
  model is ready). A daemon that cannot get them from Hugging Face (or the
  mirror in `HF_ENDPOINT`) fetches them as above (`peer_fetcher_for`),
  with 600 s per peer and in all for the 90 MB file, and checks them
  against the pins.

## Service calls

Paid service requests (`docs/services_market.md`, "How requests travel")
use `/autonet/service/1.0.0`: one JSON envelope and one JSON reply per
stream, same-stream framing (write without closing, read, close).

- **Envelope:** `{"target": <peer id>, "hops": n, "timeout": s, "msg":
  <service_request>}`. The reply is the provider's `service_request` reply.
- **Serving:** a peer that is the target (or gets no target) passes `msg`
  to its service handler. No handler: an `ok: false` "serves no services"
  reply.
- **Client (`AutonetHost.request_service`):** if not connected to the
  target, dial it first (known addresses, else the DHT; directly or through
  its relay circuit), then send direct. Only if that fails is each
  connected peer, relays first, asked to forward it with `hops: 1`, moving
  on only on a `relay_error` or a failed dial. A request is never sent down
  a second path once one reached the target. It raises `ConnectionError`
  when nothing was delivered and `TimeoutError` when a delivered request
  got no reply.
- **Forwarding:** a peer that is not the target forwards only if `hops >
  0` and it is connected to the target, with `hops: 0`, piping the reply
  back (capped at `SERVICE_RELAY_MAX_HOPS = 1`; a forwarding peer waits at
  most `service_relay_timeout`, 600 s, relay flag `--service-timeout`).
- **Who forwards.** When the dial to the target fails, only relays are asked
  to forward: a connected seed (the DAO Registry's `bootstrap` list, so it
  counts from the moment it connects) or a peer that advertised the
  capability role `relay`. Other daemons are never asked, because a
  request's args are plaintext to whoever forwards it. The client moves on
  after a `relay_error`, a failed dial, or a reply its `accept` callback
  refuses (the buyer accepts only a reply signed by the provider it paid,
  `docs/services_market.md`); all relay attempts share the caller's
  deadline.
- **Capabilities are kept under the peer that sent them** (the stream's
  Noise-authenticated peer id), not the peer id the payload names, so a
  peer can claim the relay role only for itself. Unknown capability keys
  are dropped, not fatal.

## Peer info

`/autonet/peer-info/1.0.0`: `{"peer": <id>}` in, `{"peer", "connected",
"reachable", "addrs", "roles", "timestamp"}` out, from the capability that
peer sent the answering node. A daemon asks its relays (seeds and peers
advertising the relay role) at once, to learn whether a peer it is not
connected to says it is reachable. Answers are cached for 10 minutes
(`AutonetHost.peer_reachable`, which asks in the background on a miss);
service listings show it (`reachable`).

## Service streams

`/autonet/service-stream/1.0.0` gives a buyer's and a provider's daemons a
two-way stream for a stream service (`nodes/common/service_stream.py`;
the handshake, payment and receipts are in `docs/services_market.md`,
Decision 2026-11-10 "a stream between buyer and seller"). Frames are
`[kind][4-byte length][payload]`, kind `C` (JSON control) or `D`
(application bytes).

The stream rides the mesh's connections (`AutonetHost.route_to`): one
already open to the provider; else the mesh dialer (`reach_peer`: the
provider's direct addresses first, its relay circuits racing them 2 s
later, the DHT when no address is known). Between two NAT'd daemons DCUtR
then upgrades the circuit to a direct, hole-punched TCP connection where
both NATs allow it. Nothing here opens or closes a circuit of its own.

## Proof

- `tests/test_p2p_mesh.py`: both transports, reservations and circuit
  dials, DHT provider fetch over a circuit, gossip after the relayed leg
  closes, no blob forwarding, the mesh dialer, the peer cache, the
  connection cap, SOCKS5. `tests/test_service_stream.py`: paid streams
  direct and through a relay circuit on TCP and WebSocket, megabytes each
  way, the relay's operator caps ending circuits, the payment handshake.
  `tests/test_search_model_mesh.py`: the search
  model's files provided by a NAT'd holder and fetched by a daemon that
  joins later, through the DHT record. `tests/test_evidence_mesh.py`: a
  malware flag's evidence provided by the reviewer's NAT'd daemon and read
  with `fetch_evidence` by a daemon that joins later, through the DHT
  record.
- `scripts/mesh_nat_harness.py`: Docker, two LANs behind NAT gateways
  (iptables MASQUERADE, unsolicited inbound dropped), a public relay, a
  public daemon and an HTTPS-only daemon behind a CONNECT proxy. Checks
  hole-punched NAT'd to NAT'd links (the remote end is the other NAT's
  public address), tool blobs found through the DHT, gossip and fetches
  with the relay stopped, the proxied daemon working through the relay,
  and a restart rejoining through the peer cache. Recorded output:
  `experiments/mesh_nat/harness_output.txt`.

## Not covered

- Symmetric NATs (a new port per destination, common on mobile carriers
  and some corporate networks) defeat TCP hole punching; daemons behind
  them stay on relayed connections.
- QUIC and WebRTC are off (`enable_quic`); hole punching is TCP only.
- Relayed connections run end to end Noise but cross the relay's
  bandwidth; a relay is not paid, so a jurisdiction whose daemons sit
  behind symmetric NATs depends on operators willing to run uncapped
  relays, or on providers being reachable.
- Blobs held only in the in-memory resolver map (carry bundles) are not
  persisted, so a daemon that restarts cannot serve the ones it published
  before the restart. Peers that fetched them, and relays they were
  offered to, can.
