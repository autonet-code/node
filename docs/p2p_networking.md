# P2P networking behind NAT

Most daemons run on home or cloud machines that cannot accept inbound
connections. This page covers how they still join, gossip and fetch blobs.
Code: `nodes/common/p2p.py`, `nodes/common/blob_resolver.py`,
`nodes/common/config.py`.

## Config

| Field (`p2p.*`) | Env | Meaning |
|---|---|---|
| `listen_port` | `AUTONET_P2P_PORT` | TCP port (0 = random) |
| `listen_host` | `AUTONET_P2P_HOST` | Bind address |
| `bootstrap_peers` | `AUTONET_P2P_BOOTSTRAP` | Comma list of multiaddrs (`.../p2p/<peer id>`) |
| `announce_addrs` | `AUTONET_P2P_ANNOUNCE` | Comma list of addresses peers should dial us on |
| `bootstrap_redial_interval` | | Seconds between redials of unconnected bootstrap peers (0 = dial once) |
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
  rejoins on its own.

## Seeds and relay nodes

- **Seeds come from the DAO Registry.** `jurisdiction.<id>.bootstrap` holds
  comma-separated multiaddrs, changed by proposal like the charter. A daemon
  uses them when `p2p.bootstrap_peers` (or `AUTONET_P2P_BOOTSTRAP`) is not
  set locally; local config always wins.
- **Relay nodes** (`atn relay`, a daemon mode) are the natural seeds:
  networking only, light enough for a small server. A relay keeps a stable peer id
  (`<data>/relay.key`), subscribes to each jurisdiction it serves
  (`--jurisdiction <chain_id>:<DAO>`) so gossip crosses it between peers that
  are not connected to each other, relays and caches blobs on disk, and
  answers reachability checks. No close, no chain, no agents, no model. It
  prints its full multiaddr on start: that is the address to publish.

```
atn relay --port 4001 --announce /dns4/hub.example.org/tcp/4001
```

Without `--jurisdiction` it relays the jurisdiction this install would
join; pass `--jurisdiction <chain_id>:<DAO>` (repeatable) to choose. To make
a relay a seed, the DAO writes its printed multiaddr into
`jurisdiction.<id>.bootstrap` by proposal.

## The ws transport (relays behind an HTTPS proxy)

`p2p.transport` (`AUTONET_P2P_TRANSPORT`, `atn relay --transport`) is `tcp`
(default) or `ws`. py-libp2p runs ONE transport per host, so every node of a
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

## Not covered

- Hole punching (py-libp2p ships DCUtR) and circuit relay v2 are not used. NAT'd daemons cannot
  dial each other directly; gossip and blobs go through reachable peers.
- Blobs held only in the in-memory resolver map (world checkpoints, carry
  bundles) are not persisted, so a daemon that restarts cannot serve the
  ones it published before the restart. Peers that fetched them can.
