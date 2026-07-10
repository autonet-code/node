"""Fees-only + REP-from-earnings ruleset — SIM-ONLY.

=====================================================================
SAME GROUND RULES AS v4_rules.py / v4_1_rules.py: this is NOT
production code. The "fees-only + REP-from-earnings" model was ratified
2026-07-10 to be sim-validated BEFORE any spec/build. Two structural
changes on top of v4.1 (D'/E' + supply-pegged β), each tagged
[FO-1] (pool source) / [FO-2] (rep source). Everything else — the
tool-usage mint math (household log1p damping, composition fan-out,
mint-weight scaling by rep share with ε for zero-rep, β cap) and the
review/drift/credibility machinery — is CARRIED UNCHANGED from v4.1 and
is driven through the REAL v4.1 close (`compute_v41_epoch`).
=====================================================================

What is REAL vs STUBBED (this module):

REAL (driven, not reimplemented):
  - The entire v4.1 tool-usage mint + drift + credibility close:
    `v4_1_rules.compute_v41_epoch`. We call it verbatim; we only change
    (a) what number we feed it as `emission_pool` and (b) how we accrue
    reputation from its output. The usage math the attack targets is the
    real thing.

STUBBED (thin, documented):
  - Service layer. v4.1 has no service GMV in the sim. We add a minimal
    exogenous-demand service market: providers hold a service, customers
    pay a fee-inclusive price, `FEE_RATE` (2.5%) of the payment is the
    fee, half of which BURNS into next epoch's tool-mint pool
    (`BURN_FRACTION`=0.5 → pool = 1.25% of GMV). This mirrors
    ServiceMarket.sol + payForService semantics at the granularity the
    economic question needs (no EIP-712 channels). Service pricing is
    off-chain in production too, so the fee/burn arithmetic is the load-
    bearing part and it is exact here.

The two changes:

  [FO-1] POOL = EXACTLY the fees burned that epoch. NO base pool. The
     v4.1 harness set `emission_pool = 100 (base) + recycled_fees`; here
     `emission_pool = burned_fees` and can be 0. When the pool is 0 the
     v4.1 close mints nothing (scale=0), which is correct: zero service
     volume → zero tool ATN. Tools are still FREE (usage attestations
     cost nothing) — only the POOL SIZE is now fee-derived.

  [FO-2] REP is NOT the rep-holder-attributable portion of tool mint
     (that was v4.1's D'). REP is CLAIMED 1:1 on ATN EARNINGS:
       - service providers claim REP on NET service revenue
         (gross received − any fees/burn they themselves paid as a
         customer), and
       - tool authors claim REP on their fee-pool distribution
         (their `agent_mint` from the v4.1 close).
     Pure spenders/buyers earn nothing. THIS IS THE ATTACK SURFACE: a
     dust ring that captures tool-pool ATN now claims REP on that
     capture — the D' voice-firewall (zero-rep usage grants no rep) is
     GONE by construction, because rep is downstream of *earnings*, and
     the ring earns pool ATN.

     A `rep_claim_fraction` knob (default 1.0 = full 1:1) lets us test
     partial claims. `service_rep_only=True` tests the variant where
     ONLY service revenue (not tool-pool capture) is rep-eligible — the
     obvious candidate fix if [FO-2] breaks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from harness import Agent, Tool, household_of
from v4_1_rules import Review, V41State, compute_v41_epoch, v41_rank_score

FEE_RATE = 0.025          # 2.5% of service payment volume is the fee
BURN_FRACTION = 0.5       # half the fee burns into the pool (→ 1.25% of GMV)


@dataclass
class Service:
    """A remote offering (ServiceMarket.sol). NOT a tool: no substrate
    standing, no mint of its own; it only moves ATN + generates fees. Its
    PROVIDER earns ATN and (under FO-2) claims REP on net revenue."""
    service_id: str
    provider_house: str       # household (wallet) of the provider
    price: float = 1.0        # ask price per call (fee-inclusive)


def service_fees(payments_gmv: float) -> Tuple[float, float]:
    """Given total service payment volume this epoch, return
    (fee_total, burned). fee = FEE_RATE·gmv; burned = BURN_FRACTION·fee.
    The burned half is what funds next epoch's tool-mint pool. The other
    half is DAO treasury (out of scope for the ring loop)."""
    fee = FEE_RATE * payments_gmv
    return fee, BURN_FRACTION * fee


@dataclass
class FeesOnlyState:
    """Wraps the v4.1 carry state and adds the fee/earnings ledgers the
    fees-only model needs."""
    v41: V41State = field(default_factory=V41State)
    # cumulative ATN earnings by household, split by source (for the
    # REP-from-earnings claim and the conservation asserts).
    cum_tool_earn: Dict[str, float] = field(default_factory=dict)
    cum_service_earn: Dict[str, float] = field(default_factory=dict)
    cum_rep_claimed: Dict[str, float] = field(default_factory=dict)


def run_service_market(
    *,
    payments: List[Tuple[str, str, float]],   # (customer_house, service_id, price)
    services: Dict[str, Service],
) -> Dict[str, Any]:
    """Settle one epoch of the service market.

    Each payment moves `price` ATN customer→provider. The fee is taken
    OUT of the provider's gross (theft-ceiling / labeled-transfer model:
    provider nets price·(1−FEE_RATE); fee·FEE_RATE is split burn/treasury).
    Returns per-provider gross/net, per-customer spend, GMV, fee, burn.
    """
    gmv = 0.0
    provider_gross: Dict[str, float] = {}
    provider_net: Dict[str, float] = {}
    customer_spend: Dict[str, float] = {}
    for (cust, sid, price) in payments:
        svc = services.get(sid)
        if svc is None:
            continue
        gmv += price
        prov = svc.provider_house
        fee = FEE_RATE * price
        net = price - fee
        provider_gross[prov] = provider_gross.get(prov, 0.0) + price
        provider_net[prov] = provider_net.get(prov, 0.0) + net
        customer_spend[cust] = customer_spend.get(cust, 0.0) + price
    fee_total, burned = service_fees(gmv)
    return {
        "gmv": gmv,
        "fee_total": fee_total,
        "burned": burned,
        "treasury": fee_total - burned,
        "provider_gross": provider_gross,
        "provider_net": provider_net,
        "customer_spend": customer_spend,
    }


def compute_fees_only_epoch(
    *,
    agents: Dict[str, Agent],
    state: FeesOnlyState,
    tools: Dict[str, Tool],
    usage_counts: Dict[str, Dict[str, float]],
    reviews: Dict[str, List[Review]],
    author_house: Dict[str, str],
    burned_pool: float,                # [FO-1] the fee-burn from LAST epoch
    service_settlement: Dict[str, Any],  # this epoch's run_service_market output
    delta: float,
    epsilon: float = 0.05,
    beta: Optional[float] = None,
    rep_claim_fraction: float = 1.0,
    service_rep_only: bool = False,
) -> Dict[str, Any]:
    """One fees-only close.

    Runs the REAL v4.1 tool close with `emission_pool = burned_pool`
    [FO-1], then applies REP-from-earnings [FO-2] over BOTH the tool-pool
    distribution and net service revenue. Mutates `state`.

    Returns agent_mint (tool ATN), rep_claim (voice gained this epoch by
    household), plus the v4.1 diagnostics and fee/earn breakdown.
    """
    # ---- [FO-1] the v4.1 tool close, pool = burned fees (may be 0) -----
    v41_out = compute_v41_epoch(
        agents=agents, state=state.v41, tools=tools,
        usage_counts=usage_counts, reviews=reviews,
        author_house=author_house, emission_pool=burned_pool,
        delta=delta, epsilon=epsilon, beta=beta,
    )
    tool_mint = v41_out["agent_mint"]        # household -> tool-pool ATN

    # ---- [FO-2] REP-from-earnings -------------------------------------
    # Every household's ATN earnings THIS epoch:
    #   tool_earn[h]    = its share of the fee pool (tool authorship)
    #   service_earn[h] = its NET service revenue
    # REP claimed = rep_claim_fraction * earnings (service_rep_only drops
    # the tool-pool term — the candidate fix). Pure spenders earn nothing.
    provider_net = service_settlement.get("provider_net", {})
    rep_claim: Dict[str, float] = {}
    tool_earn_epoch: Dict[str, float] = {}
    service_earn_epoch: Dict[str, float] = {}
    for h, amt in tool_mint.items():
        if amt > 0:
            tool_earn_epoch[h] = amt
    for h, amt in provider_net.items():
        if amt > 0:
            service_earn_epoch[h] = amt

    houses = set(tool_earn_epoch) | set(service_earn_epoch)
    for h in houses:
        earn = service_earn_epoch.get(h, 0.0)
        if not service_rep_only:
            earn += tool_earn_epoch.get(h, 0.0)
        claim = rep_claim_fraction * earn
        if claim > 0:
            rep_claim[h] = claim

    # accrue cumulative ledgers
    for h, amt in tool_earn_epoch.items():
        state.cum_tool_earn[h] = state.cum_tool_earn.get(h, 0.0) + amt
    for h, amt in service_earn_epoch.items():
        state.cum_service_earn[h] = state.cum_service_earn.get(h, 0.0) + amt
    for h, amt in rep_claim.items():
        state.cum_rep_claimed[h] = state.cum_rep_claimed.get(h, 0.0) + amt

    return {
        "agent_mint": tool_mint,               # tool-pool ATN (household)
        "rep_claim": rep_claim,                # [FO-2] voice gained this epoch
        "tool_earn_epoch": tool_earn_epoch,
        "service_earn_epoch": service_earn_epoch,
        "total_tool_mint": sum(tool_mint.values()),
        "burned_pool_in": burned_pool,
        "service": service_settlement,
        # carry v4.1 diagnostics
        "positions": v41_out["positions"],
        "credibility": v41_out["credibility"],
        "docked_this_epoch": v41_out["docked_this_epoch"],
        "review_mass": v41_out["review_mass"],
        "observed_zero_share": v41_out["observed_zero_share"],
        "zero_cap_scale": v41_out["zero_cap_scale"],
    }


# rank score is unchanged from v4.1
fees_only_rank_score = v41_rank_score
