# Sponsored inference (work-AI)

Status: BUILT — beta. The sponsor side (binding store, authorization,
budget) ships in `atn/sponsor_bindings.py` + `atn/autonet_service.py`; the
dependent side is the RPB provider (`atn/providers/rpb.py`), resolved in
`atn/runtime/provider_manager.py`. The **RPB Network** card in the app's AI
Input tab is the owner-facing surface.

Wire surface: `get_my_sponsor` / `set_my_sponsor` (dependent side),
`create_sponsor_agent` / `list_sponsor_bindings` / `update_sponsor_budget` /
`remove_sponsor_binding` (sponsor side).

This is the doc for the **owner** on either end: someone lending inference
capacity, or someone running a daemon on somebody else's.

## What it is

A *sponsor* daemon supplies LLM inference to a *dependent* daemon over the
peer-to-peer network. The sponsor pays for the tokens — with its own API key
or subscription — and the dependent's agents run without any provider
credentials of their own.

The sponsor is the resource owner, so the sponsor is the authority. It
decides who it serves and how many tokens each may spend. Nothing the
dependent configures can grant itself access.

## The dependent is a wallet, not an agent

**One rule: a dependent is an owner wallet — one 0x address per daemon.**

The sponsor binds that address and stops caring what happens behind it. How
many agents the dependent runs, how deeply they nest, whether any of them is
registered on-chain — none of it is visible to the sponsor, and none of it
needs to be. Everything on that daemon presents as the one address and draws
on the one budget.

This is deliberate. The alternative — binding individual agents — means the
sponsor re-binds every time the dependent creates an agent, and it makes
sponsorship depend on each agent having its own on-chain registration. The
household is the natural unit: you sponsor a *person's daemon*, not a
shifting set of processes inside it.

Consequences worth knowing:

- **A daemon with no owner wallet cannot be a dependent.** There is no
  identity to bind. (Same principle as tool publishing — see
  `docs/tool_substrate.md`, `Decision (2026-07-24)`: no claimable identity,
  no participation beyond the private plane.)
- **Every agent on a sponsored daemon shares one budget.** A runaway agent
  spends the same grant as a careful one. The token budget is the sponsor's
  only ceiling, so set one.
- **Sponsorship is configured once, per daemon** — `autonet.sponsor_address`
  in `config.yaml`, surfaced in the RPB Network provider card. There is no
  per-agent sponsor setting. (`AgentDefinition.sponsor_address` survives as a
  dead field so existing `agent.yaml` files still load; nothing reads it.)

## One sponsor

A dependent daemon has **at most one sponsor**. Not a ranked list, no
failover to a second sponsor when a budget runs dry, no mixing a
household-level grant with per-agent ones.

This is a deliberate limit, not an oversight. Multi-sponsor raises questions
with no obvious answer — which sponsor serves a given request, what happens
when one budget empties, whether an agent-specific grant overrides a
household one — and every answer is easy to add later but hard to withdraw
once daemons depend on it. No one has hit the limit yet, so the limit stays.

## The two sides

### Sponsoring someone (the paying end)

Turn on sponsor mode and the daemon advertises, over gossip, that it will
proxy inference. Configure which provider and model serve those requests, or
leave them empty to reuse the same resolution local agents get.

Then bind each dependent by its owner wallet address, with a token budget.
A budget of 0 means unlimited — serve until the binding is removed.

Authorization is a single check: the requesting address must be in the
binding store, and its budget must not be exhausted. An unbound address is
refused with `not an authorized dependent`. There is no discovery, no
trust-on-first-use, and no way for a dependent to add itself.

### Being sponsored (the consuming end)

Set the sponsor's 0x address in the RPB Network provider, and point an agent
at the `rpb` provider. Requests route to that sponsor, carrying the
dependent's owner wallet as the identity.

Two things follow from naming a sponsor:

- **Routing is bound, not market-discovered.** Only that sponsor is
  considered — the dependent will not silently shop for another.
- **The sponsor dictates the model.** When a sponsor is named, its
  advertised model is accepted regardless of what the agent asked for. The
  employer chooses the tool.

Leaving the sponsor address empty is *not* "off": the dependent falls back to
open discovery and will use any sponsor it finds that matches the requested
model, preferring lower latency. Set the address if you mean to be bound to
one.

## Budget

Budgets are denominated in **tokens** (input + output), tracked by the
sponsor. Each response carries the dependent's remaining balance
(`remaining_budget_tokens`), so the dependent learns what is left without
asking the sponsor separately; the provider logs a warning below 10k
remaining.

ATN settlement is not wired to this yet — v1 is a token allowance, not a
payment. The economic rails (`payForService`, the venture loop) are separate
and documented in `docs/services_market.md`.

When a budget is exhausted the sponsor refuses with `budget exhausted`.
There is no failover; the dependent's agents stop until the sponsor raises
the budget.

## Why the sponsor holds the state

Every piece of authorization lives on the sponsor's disk
(`sponsor_bindings.json`) and nowhere else. The dependent holds no token, no
grant, no credential. It presents an address; the sponsor decides.

This means a compromised or malicious dependent can do exactly one thing:
spend a budget the sponsor already agreed to. It cannot widen its grant,
reach another sponsor's capacity, or impersonate a different dependent
without that dependent's wallet key.

## See also

- `docs/providers.md` — inference providers generally, including RPB
- `docs/two_plane_inference.md` — substrate retrieval paired with an LLM
- `docs/services_market.md` — the paid remote-API rail (distinct from
  sponsorship: services are sold, sponsorship is granted)
