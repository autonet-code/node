"""TrialRunner — service verifier trials (venture vault rail).

Design: ``docs/tool_substrate.md`` (Verifier trials) + the venture-vault
tokenomics memo. A venture's value terminates in a MOAT (credentials,
data, hardware, state) — which, unlike pinned tool code, cannot be READ.
So it is PROBED: a validator agent runs the venture's own pre-committed
black-box trial battery against the live service and attests what it
observed. This is the vetting pipeline generalized from "read the code"
to "exercise the endpoint".

The daemon-facing flow (this module):

  1. FETCH the venture prospectus by digest (blob store first, then the
     libp2p blob rail — same digest-verified path ToolStore.vet uses).
  2. EXECUTE each declared trial case against the service's MCP surface
     via an injected transport seam. Live service-invocation plumbing is
     deliberately thin right now, so the transport is a callable
     (endpoint, op, args) -> result-dict, wired to the WS service-request
     client in production and to a fake in tests. The prospectus's own
     free-inference credentials/endpoint are threaded through.
  3. SCORE pass/fail per the prospectus's OWN pre-committed criteria
     (the whole point of a black-box trial: the venture commits to the
     bar before any validator runs it).
  4. BLOB-STORE a trial report (reviewable evidence a later dispute
     argues against) and RETURN the verdict + report digest + the
     ``attestTrial`` calldata the owner surface submits on-chain.

The on-chain ``VentureVault.attestTrial`` tx is built in parallel by the
contracts workstream; this module returns the calldata (no generic
contract-call seam exists on OnChainService yet — the pattern there is
per-method builders), so the owner surface can submit it once the vault
is deployed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .runtime import Runtime

log = logging.getLogger(__name__)

PROSPECTUS_KIND = "venture_prospectus"

# Transport seam: one service invocation. Wired to the WS service-request
# client in production; a fake in tests. Returns the service's result
# dict ({"result": ...} on success, {"error": ...} on failure) — the same
# shape ToolStore.call returns, so scoring is uniform across transports.
Transport = Callable[[str, str, dict[str, Any]], Any]


def is_prospectus(payload: Any) -> bool:
    return (isinstance(payload, dict)
            and payload.get("kind") == PROSPECTUS_KIND)


class TrialRunner:
    """Executes venture prospectus trial batteries against a service.

    The runner owns fetch + execute + score + report; the on-chain
    attestation is the owner's to submit (calldata returned). It leans on
    ``runtime.tool_store`` for the digest-verified blob rail (fetch +
    store) so there is one content-addressed cache on the daemon.
    """

    def __init__(self, runtime: "Runtime",
                 transport: Optional[Transport] = None) -> None:
        self._runtime = runtime
        # Injected in production (WS service-request client); a fake in
        # tests. Absent → trials that need a live call fail loud rather
        # than silently pass.
        self.transport: Optional[Transport] = transport

    # ------------------------------------------------------------------
    # Public flow
    # ------------------------------------------------------------------

    async def run_trial(self, caller: str,
                        prospectus_digest: str) -> dict[str, Any]:
        """Fetch → execute → score → report → calldata.

        Returns ``{"prospectus_digest", "service_digest", "verdict"
        ("pass"|"fail"), "passed", "total", "cases": [...],
        "report_digest", "attest_calldata"}`` or ``{"error": ...}``.
        """
        digest = (prospectus_digest or "").strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return {"error": "prospectus_digest must be the 64-hex artifact "
                             "digest of a published venture_prospectus"}

        store = self._runtime.tool_store
        payload = await store.fetch_payload(digest)
        if not is_prospectus(payload):
            return {"error": f"cannot resolve a venture_prospectus at "
                             f"{digest[:16]} (not fetchable / wrong kind)"}

        battery = payload.get("battery")
        if not isinstance(battery, list) or not battery:
            return {"error": "prospectus carries no trial battery "
                             "('battery' must be a non-empty list of cases)"}

        service_digest = str(payload.get("service_digest") or "")
        # Endpoint + free-inference credentials the venture funds so the
        # validator can probe at no personal cost (author-funded trials).
        endpoint = str(payload.get("endpoint") or "")
        credentials = payload.get("credentials") or {}

        cases: list[dict[str, Any]] = []
        passed = 0
        for i, case in enumerate(battery):
            outcome = await self._run_case(i, case, endpoint, credentials)
            if outcome["passed"]:
                passed += 1
            cases.append(outcome)

        total = len(battery)
        # Greenlight bar: the prospectus PRE-COMMITS the pass threshold
        # (fraction 0..1). Default: all cases must pass (a moat that only
        # sometimes works is a fail). The venture set the bar, not us.
        threshold = payload.get("pass_threshold")
        try:
            threshold = float(threshold) if threshold is not None else 1.0
        except (TypeError, ValueError):
            threshold = 1.0
        threshold = max(0.0, min(1.0, threshold))
        pass_rate = (passed / total) if total else 0.0
        verdict = "pass" if pass_rate >= threshold else "fail"

        report_digest = self._store_report(
            caller, digest, service_digest, verdict,
            passed, total, threshold, cases)

        attest_calldata = self._build_attest_calldata(
            digest, report_digest, verdict == "pass")

        return {
            "prospectus_digest": digest,
            "service_digest": service_digest,
            "verdict": verdict,
            "passed": passed,
            "total": total,
            "pass_rate": round(pass_rate, 6),
            "pass_threshold": threshold,
            "cases": cases,
            "report_digest": report_digest,
            "attest_calldata": attest_calldata,
            "note": ("submit attest_calldata via the owner on-chain surface "
                     "to record this trial on VentureVault.attestTrial"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_case(self, idx: int, case: dict[str, Any],
                        endpoint: str,
                        credentials: dict[str, Any]) -> dict[str, Any]:
        """Execute one declared trial case and score it against the
        prospectus's own criteria.

        A case is ``{"op"?, "args", "expect_digest" | "expect_error" |
        "expect_contains"}``. Scoring mirrors evidence replay: an
        ``expect_error`` case passes iff the call errors; an
        ``expect_digest`` case passes iff the canonical result digest
        matches; an ``expect_contains`` case passes iff the JSON-serialized
        result contains the substring. A malformed case fails loud (never
        silently passes)."""
        if not isinstance(case, dict):
            return {"case": idx, "passed": False,
                    "error": "case is not an object"}
        op = str(case.get("op") or "")
        args = case.get("args")
        if not isinstance(args, dict):
            args = {}
        # Thread the venture-funded credentials into the call args under a
        # reserved key the service expects (author-funded free inference).
        if credentials:
            args = {**args, "_credentials": credentials}

        if self.transport is None:
            return {"case": idx, "passed": False,
                    "error": "no service transport wired — cannot probe a "
                             "live service surface"}

        try:
            result = await _maybe_await(self.transport(endpoint, op, args))
        except Exception as exc:
            result = {"error": f"transport raised: {exc}"}
        if not isinstance(result, dict):
            result = {"result": result}

        errored = "error" in result

        if "expect_error" in case:
            passed = errored
            return {"case": idx, "passed": passed, "errored": errored,
                    "observed_error": (str(result.get("error"))[:1000]
                                       if errored else None)}

        if errored:
            # A case that expected a result but the call errored: fail.
            return {"case": idx, "passed": False, "errored": True,
                    "observed_error": str(result.get("error"))[:1000]}

        observed_digest = self._result_digest(result)
        if "expect_digest" in case:
            passed = observed_digest == str(case.get("expect_digest"))
            return {"case": idx, "passed": passed,
                    "observed_digest": observed_digest,
                    "expect_digest": str(case.get("expect_digest"))}
        if "expect_contains" in case:
            blob = json.dumps(result.get("result"), sort_keys=True,
                              default=str)
            passed = str(case.get("expect_contains")) in blob
            return {"case": idx, "passed": passed,
                    "observed_digest": observed_digest,
                    "expect_contains": str(case.get("expect_contains"))}

        # No criterion declared: a successful, non-erroring call is the
        # weakest possible bar — record it, mark passed (liveness only).
        return {"case": idx, "passed": True,
                "observed_digest": observed_digest,
                "note": "no criterion declared; scored liveness only"}

    @staticmethod
    def _result_digest(result: dict[str, Any]) -> str:
        """Canonical sha256 of a service result's ``{"result": ...}``
        payload — the SAME canonicalization ToolStore evidence replay
        uses, so prospectus authors compute expect_digest identically."""
        payload = result.get("result")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True,
                       default=str).encode("utf-8")).hexdigest()

    def _store_report(self, caller: str, prospectus_digest: str,
                      service_digest: str, verdict: str, passed: int,
                      total: int, threshold: float,
                      cases: list[dict[str, Any]]) -> str:
        """Blob-store the trial report (reviewable evidence). Best-effort:
        a standalone install without the blob store returns ''."""
        report = {
            "kind": "venture_trial_report",
            "prospectus_digest": prospectus_digest,
            "service_digest": service_digest,
            "validator": self._consensus_identity(caller),
            "verdict": verdict,
            "passed": passed,
            "total": total,
            "pass_threshold": threshold,
            "cases": cases,
            "ts": int(time.time()),
        }
        try:
            return self._runtime.tool_store._blob_store().add_json(report) or ""
        except Exception as exc:
            log.warning("trial report blob failed: %s", exc)
            return ""

    def _build_attest_calldata(self, prospectus_digest: str,
                               report_digest: str, passed: bool) -> str:
        """ABI-encode ``VentureVault.attestTrial(bytes32 prospectus,
        bytes32 report, bool passed)`` calldata for the owner to submit.

        Encoded WITHOUT web3/a live node — the selector is keccak of the
        signature and the three 32-byte args are packed directly, so this
        works on a daemon with no RPC configured (the owner submits it
        wherever the vault is deployed). Returns '' if keccak is
        unavailable (never blocks the verdict)."""
        try:
            selector = _keccak_selector(
                "attestTrial(bytes32,bytes32,bool)")
        except Exception as exc:
            log.debug("attestTrial calldata skipped (no keccak): %s", exc)
            return ""
        arg_prospectus = _hex_to_bytes32(prospectus_digest)
        arg_report = _hex_to_bytes32(report_digest)
        arg_passed = (b"\x00" * 31) + (b"\x01" if passed else b"\x00")
        return "0x" + (selector + arg_prospectus + arg_report
                       + arg_passed).hex()

    def _consensus_identity(self, caller: str) -> str:
        try:
            return self._runtime.tool_store._consensus_identity(caller)
        except Exception:
            return caller or "user"


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, else return it. Lets the
    transport be either sync or async."""
    if hasattr(value, "__await__"):
        return await value
    return value


def _hex_to_bytes32(hex_str: str) -> bytes:
    """A 64-hex digest → 32 bytes (right-padded/truncated). Empty → zeros."""
    clean = (hex_str or "").removeprefix("0x")
    try:
        raw = bytes.fromhex(clean.ljust(64, "0")[:64])
    except ValueError:
        raw = b"\x00" * 32
    return raw


def _keccak_selector(signature: str) -> bytes:
    """First 4 bytes of keccak256(signature) — the function selector."""
    try:
        from eth_utils import keccak  # ships with web3/eth-account
        return keccak(text=signature)[:4]
    except ImportError:
        from Crypto.Hash import keccak as _k  # pycryptodome fallback
        h = _k.new(digest_bits=256)
        h.update(signature.encode("utf-8"))
        return h.digest()[:4]
