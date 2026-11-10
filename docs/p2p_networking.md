# P2P networking behind NAT

Most daemons run on home or cloud machines that cannot accept inbound
connections. This page covers how they still join, gossip, fetch blobs and
open service streams to each other. Code: `nodes/common/p2p.py`,
`nodes/common/blob_resolver.py`, `nodes/common/circuit_relay.py`,
`nodes/common/service_stream.py`, `nodes/common/config.py`.

## Config

| Field (`p2p.*`) | Env | Meaning |
|---|---|---|
| `listen_port` | `AUTONET_P2P_PORT` | TCP port (0 = random) |
| `listen_host` | `AUTONET_P2P_HOST` | Bind address |
| `bootstrap_peers` | `AUTONET_P2P_BOOTSTRAP` | Comma list of multiaddrs (`.../p2p/<peer id>`) |
| `announce_addrs` | `AUTONET_P2P_ANNOUNCE` | Comma list of addresses peers should dial us on |
| `bootstrap_redial_interval` | | Seconds between redials of unconnected bootstrap peers (0 = dial once) |
| `seed_refresh_interval` | | Seconds between re-reads of the Registry seed list on a running daemon (default 300, 0 = read at join only) |
| `enable_upnp` | | Map the listen port on the home router (default on) |
| `auto_reach` | | Find and announce our public address when no `announce_addrs` is set (default on) |
| `reach_interval` | | Seconds between reachability checks (default 300) |

## Reachable and NAT'd daemons

Home users need no network setup. With no `announce_addrs`:

1. **UPnP:** the daemon asks the home router to forward its listen port
   (most routers support it; some ship with it off) and renews the
   mapping every check.
2. **Reach check** (`/autonet/reach/1.0.0`): the daemon asks a connected
   peer what IP it sees it on and whether a fresh TCP connect to that IP
   and our listen port works. The peer dials only the requester's
   observed IP. If it works and the IP is public, the daemon announces
   `/ip4/<ip>/tcp/<port>`; if a later check fails, it goes back to its
   listen addresses. This also covers a manual port forward, and repeating
   the check every `reach_interval` follows a changed home IP with no DNS.
3. Otherwise the daemon stays NAT'd and keeps working by dialling out.

Whether a daemon announces a dialable address is advertised in its
capability (`NodeCapability.reachable`: configured announce addresses, or
one the reach check confirmed). Buyers dial a reachable provider directly;
a NAT'd one is reached through a relay. The app shows it to the owner
(`list_services` returns `reachability`, with advice when the daemon is not
reachable), and catalog rows show it per provider (`reachable`).

- A **reachable** daemon (public IP, port forward or tunnel) sets
  `announce_addrs` to its public address. That address then replaces the
  listen addresses in identify, the signed peer record and the gossiped
  capability. The listen socket is unchanged.
- A **NAT'd** daemon sets no announce address and only dials out. Peers
  reach it over the connection it opened.
- `/dns4`, `/dns6` and `/dns` names in bootstrap and announce addresses are
  resolved to `/ip4` or `/ip6`, because py-libp2p's TCP transport dials
  only IP addresses. They are resolved again every
  `bootstrap_redial_interval` seconds, so a reachable daemon should
  announce a name (dynamic DNS or a stable tunnel hostname), not a bare IP.
- **IP changes.** A NAT'd daemon whose IP changes loses its connections
  and redials its bootstrap peers; nobody needs its new address. A
  reachable daemon whose announce name now resolves elsewhere advertises
  the new address to its peers at once, and peers that bootstrap to it
  redial the name's current address (stale cached addresses are dropped
  first).
- Bootstrap peers are redialled every `bootstrap_redial_interval` seconds
  while unconnected, so a daemon that started first, or lost its link,
  rejoins on its own. A daemon with no peer at all redials them at once,
  then with backoff (5 s doubling to 2 min), even with that interval at 0.

## Seeds and relay nodes

- **Seeds come from the DAO Registry.** `jurisdiction.<id>.bootstrap` holds
  comma-separated multiaddrs, changed by proposal like the charter. A daemon
  uses them when `p2p.bootstrap_peers` (or `AUTONET_P2P_BOOTSTRAP`) is not
  set locally; local config always wins. A running daemon re-reads the key
  every `seed_refresh_interval` seconds (latest block, like the join read)
  and dials seeds it is not connected to, so a relay the DAO adds reaches
  running daemons without a restart. A failed read or an emptied key keeps
  the seeds it has.
- **Relay nodes** (`atn relay`, a daemon mode) are the natural seeds:
  networking only, light enough for a small server. A relay keeps a stable peer id
  (`<data>/relay.key`), subscribes to each jurisdiction it serves
  (`--jurisdiction <chain_id>:<DAO>`) so gossip crosses it between peers that
  are not connected to each other, relays and caches blobs on disk,
  keeps the state blobs daemons offer it after each close (anchor payload,
  carry-over, agent mints, over
  `/autonet/blob-offer/1.0.0`; it fetches what it lacks from the offerer),
  so a joiner can install the anchored state when every daemon that
  computed it is offline (issue #44), forwards service calls one hop,
  relays connections between daemons connected to it (circuit relay v2,
  below), and answers reachability checks. No close, no chain, no agents,
  no model. It prints its full multiaddr on start: that is the address to
  publish.

```
atn relay --port 4001 --announce /dns4/hub.example.org/tcp/4001
```

Relayed connections are unlimited by default (a relay gives bandwidth the
way a torrent seed does; relays are not paid). Its operator may cap them:

| Flag | Env | Meaning (0 = no limit) |
|---|---|---|
| `--circuit-max-duration` | `AUTONET_RELAY_CIRCUIT_MAX_DURATION` | Seconds one relayed connection may stay open |
| `--circuit-max-bytes` | `AUTONET_RELAY_CIRCUIT_MAX_BYTES` | Bytes one relayed connection may carry, both ways together |
| `--max-circuits` | `AUTONET_RELAY_MAX_CIRCUITS` | Relayed connections open at once |
| `--no-circuits` | `AUTONET_RELAY_NO_CIRCUITS` | Relay no connections at all |

Circuit relay v2 stacks default to tight limits (go-libp2p: 2 minutes and
128 KiB per relayed connection; py-libp2p 0.6: 1 GiB per reservation and
four reservations in all), which would end any stream within minutes; none
of those apply here.

Without `--jurisdiction` it relays the jurisdiction this install would
join; pass `--jurisdiction <chain_id>:<DAO>` (repeatable) to choose. To make
a relay a seed, the DAO writes its printed multiaddr into
`jurisdiction.<id>.bootstrap` by proposal.

## The ws transport (relays behind an HTTPS proxy)

`p2p.transport` (`AUTONET_P2P_TRANSPORT`, `atn relay --transport`) is `ws`
(default) or `tcp`. py-libp2p runs ONE transport per host, so every node of a
network must use the same one.

- **Hiding a relay's address.** A `ws` relay can sit behind an HTTPS proxy
  such as Cloudflare: the relay serves plain WebSocket on the origin, the
  proxy terminates `wss` on 443, and the published seed is
  `/dns4/<relay hostname>/tcp/443/wss/p2p/<peer id>`. Peers see the proxy's
  address, not the server's. Firewall the origin to the proxy's ranges.
- **HTTPS-only machines.** A `ws` daemon dials through `HTTP_PROXY` /
  `HTTPS_PROXY`: SOCKS5 natively, HTTP CONNECT via
  `nodes/common/ws_proxy.py`. The proxy resolves names, so the dialer needs no
  DNS of its own (cloud sandboxes, corporate networks).
- **Names stay names.** In `ws` mode `/dns4` addresses are not resolved
  locally: TLS (SNI) and the HTTP Host header must carry the hostname for the
  proxy in front of a relay to route the connection.
- **Peer authentication is libp2p's,** not the TLS certificate's: the
  connection is still authenticated by the peer id (Noise), whoever
  terminates TLS.

## Blob fetch

- A resolver miss asks every connected peer and every peer known from
  capability gossip at once. The first answer whose sha256 matches wins,
  and the other requests are cancelled.
- **One-hop relay:** a request frame is `<hash>` or `<hash>|<hops>`. A peer
  that misses forwards a request with hops left to its own connected peers
  (minus the requester), with hops reduced by one, and caches what it gets
  back. A bare hash allows one hop, which is also the maximum
  (`BLOB_RELAY_MAX_HOPS`). This lets two NAT'd daemons exchange blobs
  through a reachable peer they are both connected to.
- The daemon's on-disk tool and service blob stores are served as sources
  on the host's single `/autonet/blob/1.0.0` handler, so manifests and code
  a daemon published stay fetchable after it restarts.

## Service calls

Paid service requests (`docs/services_market.md`, "How requests travel")
use `/autonet/service/1.0.0`: one JSON envelope and one JSON reply per
stream, same-stream framing (write without closing, read, close).

- **Envelope:** `{"target": <peer id>, "hops": n, "timeout": s, "msg":
  <service_request>}`. The reply is the provider's `service_request` reply.
- **Serving:** a peer that is the target (or gets no target) passes `msg`
  to its service handler (`set_service_handler`; the daemon installs the
  WS bridge's shared handler, bridged trio to asyncio). No handler: an
  `ok: false` "serves no services" reply.
- **Forwarding:** a peer that is not the target forwards only if `hops >
  0` and it is connected to the target, sending the envelope on with
  `hops: 0` and piping the reply back. Otherwise it answers
  `{"relay_error": ...}` at once. `hops` is capped at
  `SERVICE_RELAY_MAX_HOPS = 1` and a missing `hops` means 0, so a request
  travels at most one hop and a relay never fans it out.
- **Client (`AutonetHost.request_service`):** direct when connected to the
  target; else each connected relay in turn with `hops: 1`. A relay is a
  connected seed (the DAO Registry's `bootstrap` list, so it counts from
  the moment it connects) or a peer that advertised the capability role
  `relay`; other daemons are never asked, because a request's args are
  plaintext to whoever forwards it. The client moves on after a
  `relay_error`, a failed dial, or a reply its `accept` callback refuses
  (the buyer accepts only a reply signed by the provider it paid,
  `docs/services_market.md`); all relay attempts share the caller's
  deadline. It raises `ConnectionError` when nothing was delivered and
  `TimeoutError` when a delivered request got no reply. Connect limit 15 s;
  reply limit is the caller's (default 300 s); a forwarding peer waits at
  most `service_relay_timeout` (600 s; relay flag `--service-timeout`).
- **Capabilities are kept under the peer that sent them** (the stream's
  Noise-authenticated peer id), not the peer id the payload names, so a
  peer can claim the relay role only for itself. Unknown capability keys
  are dropped, not fatal.

## Peer info

`/autonet/peer-info/1.0.0`: `{"peer": <id>}` in, `{"peer", "connected",
"reachable", "addrs", "roles", "timestamp"}` out, from the capability that
peer sent the answering node. A daemon asks its relays (seeds and peers
advertising the relay role) at once, to learn whether a peer it is not
connected to says it is reachable and at which addresses, and which relay
is connected to it. Answers are cached for 10 minutes
(`AutonetHost.peer_reachable`, which asks in the background on a miss).

## Relayed connections (circuit relay v2)

Two daemons connected to the same relay get a libp2p connection to each
other through it (`nodes/common/circuit_relay.py`). The dialer sends the
relay a HOP CONNECT naming the target; the relay opens a STOP stream to
the target and copies bytes between the two streams; the two ends run
Noise and yamux over them. The relay sees only ciphertext, and each end
has authenticated the other's peer id.

- Every host accepts relayed connections (py-libp2p's own STOP handler).
  Only a host given `circuit_relay_caps` (`atn relay`) relays them, and
  only to a target already connected to it: a relay never dials for a
  CONNECT.
- The HOP side is ours, not py-libp2p 0.6's `CircuitV2Protocol`: that one
  writes each read of up to 65535 bytes to the other stream in one write,
  which Noise refuses (its message limit is 65535 bytes and yamux adds a
  12-byte header), so any relayed transfer above about 64 KiB breaks. It
  also wants the dialer to hold a reservation, accepts four reservations in
  all, caps each at 1 GiB and keys circuits by peer, so one buyer's closing
  circuit closes another's to the same provider. Ours copies in 65000-byte
  writes, needs no reservation (a RESERVE is answered OK), and applies the
  operator's caps per circuit.
- Wire: py-libp2p 0.6's circuit v2 messages (protobufs without a length
  prefix). It does not interoperate with go-libp2p or rust-libp2p relays,
  which length-prefix, nor with py-libp2p 0.8 (whose target must hold a
  reservation).
- Proven on both transports (`tcp`, `ws`) with megabytes each way
  (`tests/test_service_stream.py`).

## Service streams

`/autonet/service-stream/1.0.0` gives a buyer's and a provider's daemons a
two-way stream for a stream service (`nodes/common/service_stream.py`;
the handshake, payment and receipts are in `docs/services_market.md`,
Decision 2026-11-10 "a stream between buyer and seller"). Frames are
`[kind][4-byte length][payload]`, kind `C` (JSON control) or `D`
(application bytes).

The path, in the owner's order (`AutonetHost.route_to`):

1. a connection already open to the provider;
2. a direct dial to the addresses the provider advertises as reachable
   (its capability, or its relays' answers);
3. a relayed connection through a relay that is connected to it (relays
   that answered "connected" first).

Hole punching would sit between 2 and 3; it is not tried (below). A
relayed connection the buyer opened for a stream is closed when no stream
uses it.

## Not covered

- **Hole punching (DCUtR) is not used.** py-libp2p 0.6 (the version
  pinned, `libp2p>=0.6.0,<0.7`) ships `DCUtRProtocol`, but it cannot punch
  a NAT: it exchanges the host's listen addresses instead of the addresses
  peers observe, it dials with `host.connect`, which returns at once while
  the relayed connection exists (so no direct dial happens), and it reports
  success by checking the peer's addresses in the peerstore rather than
  the connection's transport, so it reports success over the relayed
  connection. Its TCP transport dials from an ephemeral port (no
  `SO_REUSEPORT`, no source bind), and TCP hole punching needs both sides
  to dial from the port the other dials. Running it would fake a direct
  connection, so it is off.
- **What it would take.** py-libp2p 0.8.1 (tried in a separate venv; its
  own circuit relay carried 5 MB both ways on `tcp` and `ws` once the
  target reserved) has a DCUtR written for TCP: observed addresses,
  source-bound dials with `SO_REUSEPORT`, closing the relayed connection
  after the upgrade. It was read, not run against a real NAT (localhost
  cannot show a punch). Its WebSocket transport still dials from an
  ephemeral port (`dial_with_source` exists only for TCP), so on `ws`, the
  daemon's default, hole punching would still not work; a `tcp` network
  could use it. Upgrading means moving off `libp2p.tools.async_service` (renamed to
  `anyio_service` in 0.8, which breaks `nodes/common/p2p.py`'s imports),
  re-checking the py-libp2p quirks this host works around, and moving the
  relay to 0.8's wire (length-prefixed messages, a reservation by the
  target). For `ws` it also needs a source-bound dial in the WebSocket
  transport, upstream or in a fork. Until then a NAT'd buyer reaches a
  NAT'd provider only through a relay.
- Relayed connections run end to end Noise but cross the relay's
  bandwidth; a relay is not paid, so a busy jurisdiction depends on
  operators willing to run uncapped relays, or on providers being
  reachable.
- Blobs held only in the in-memory resolver map (carry bundles) are not
  persisted, so a daemon that restarts cannot serve the
  ones it published before the restart. Peers that fetched them can.
