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

## Reachable and NAT'd daemons

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

- Circuit relay v2 (py-libp2p has it) is not used. NAT'd daemons cannot
  dial each other directly; gossip and blobs go through reachable peers.
- Blobs held only in the in-memory resolver map (world checkpoints, carry
  bundles) are not persisted, so a daemon that restarts cannot serve the
  ones it published before the restart. Peers that fetched them can.
