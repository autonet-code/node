# Head-to-head: text-claim substrate vs tool substrate

Purpose: the pre-merge comparison requested 2026-07-04 — what the
pre-tool substrate promised, what the tool substrate delivers, and what
merging `tool-substrate` into master does and does not foreclose.

## The two propositions

**A. Text-claim substrate (master, pre-branch).** The claim graph as a
priced belief structure. Its distinctive promises:

- *Equilibration as inference* — the geometric propagation kernel
  (incl. the Lindblad continuous variant): score flows through graph
  structure, so standing is not a tally but an equilibrium of the
  whole neighborhood.
- *Extrapolation from weak edges* — many individually-insignificant
  signals combine through propagation into confident standing;
  regions of the graph the network barely touched still inherit
  priced belief from their surroundings.
- *Generality* — anything expressible as a claim can be placed on the
  charter axes and debated: judgments, procedures, alignment calls.

**B. Tool substrate (this branch).** The capability commons. Its
distinctive deliveries:

- *Executable ground truth* — a CON is a failing invocation, replayable
  forever against a pinned digest. Debates terminate.
- *Revealed-preference usefulness* — attested usage (per-work-item
  cognitive attestations) instead of stated opinion; the coverage
  atlas maps what the network can do and where.
- *A closed economic loop* — author → mint → wallet → spend, plus the
  services market for what can't ship as code. Immediate user value:
  a growing library of runnable capability, not a corpus of text.

## The evidence as it stands

| Question | Text substrate | Tool substrate |
|---|---|---|
| Does debate beat counting? | phase8: +0.127 vs vote-count, Holm p=0.054 — **missed the pre-registered 0.25 bar**; demoted to opt-in kernel | Not yet contested at equivalent rigor — but CONs carry reproducible evidence, so the mechanism is structurally stronger, not just hopefully stronger |
| Retrieval value | Graph-as-index **lost** to plain embeddings (May 2026); verdicts-in-prompt **hurt** (−0.28) | Two-plane retrieval validated (+0.367); density-over-attested-coverage designed to resist SEO |
| Ground truth | None — a CON is more prose | Failing invocation, replayable (pinned class) |
| Incentive gameability | Farmable (defender loop still unbuilt) | Sim-swept: combo damper drives wash ROI ≤ 0; attestation cost is the floor price |
| Product surface | Indirect (context for LLMs) | Direct (tools run; services sell) |
| Unproven upside | **phase9**: deep contested graphs are where propagation *should* shine; phase8 only tested a depth-1 forest | Capability-gap pricing over the atlas; cross-daemon adoption dynamics at scale |

## The structural finding: these are orthogonal axes

The head-to-head dissolves on inspection. The two propositions answer
different questions:

- The tool substrate changes **WHAT is judged** (artifact kind:
  manifests + receipts instead of prose work units).
- Equilibration vs ledger changes **HOW standing is computed** (pricing
  kernel over the same graph).

Nothing in this branch touched the kernel choice. `pricing="equilibrated"`
survives intact, the charter world is unchanged, tool claims live in the
same graph equilibration would run over. If phase9 clears its
pre-committed bar on deep contested graphs, the equilibrated kernel
returns as the standing computation — *for tool claims too*, where its
weak-edge promise has a concrete new referent: cold-start tools with
sparse attestations are exactly "weak edges," and propagation from
neighboring judged tools would price them better than a bare ledger
tally can. The quantum-inference promise doesn't compete with the tool
substrate; the tool substrate is the strongest content that promise has
ever had to work on.

## What merging does and does not foreclose

Merging `tool-substrate` → master:

- does NOT delete or demote anything not already demoted by phase8's
  own pre-registered gate (ratified before this branch existed);
- does NOT preempt phase9 — the experiment doc, the equilibrated
  kernel, and the `pricing` switch are all intact and the test remains
  pre-committed-as-final;
- DOES commit the network's flagship narrative and near-term dev
  effort to capability-over-corpus: tools as the primary substrate
  item, services as the market rail.

## Recommendation

Merge, and schedule phase9 as the standing obligation it already is.
The honest statement of the bet: we moved the substrate onto content
with ground truth because the content without it missed its own bar;
the propagation kernel keeps its pre-registered chance to win back the
pricing role on the new content. If phase9 clears, we get both halves
of the original dream — quantum inference over a substrate of things
that provably work.
