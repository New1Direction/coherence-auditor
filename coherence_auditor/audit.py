"""Orchestration: belief set in, structured audit report out.

The report is plain dataclasses + a `to_dict()` so it serializes cleanly for the
MCP tool, logs, or a downstream agent. `total_exposure` is the sum of guaranteed
losses across all books found — a single scalar "how exploitable is this model's
belief set" number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import checks, dutchbook
from .checks import CheckResult
from .dutchbook import Certificate
from .elicit import Elicitor
from .propositions import BeliefSet, Proposition


@dataclass
class AuditReport:
    context: str
    results: list[CheckResult] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    belief: dict = field(default_factory=dict)

    @property
    def coherent(self) -> bool:
        return all(r.coherent for r in self.results)

    @property
    def total_exposure(self) -> float:
        return round(sum(c.guaranteed_loss for c in self.certificates
                         if c.is_book()), 12)

    def to_dict(self) -> dict:
        return {
            "coherent": self.coherent,
            "total_exposure": self.total_exposure,
            "context": self.context,
            "belief": self.belief,
            "checks": [asdict(r) | {"coherent": r.coherent} for r in self.results],
            "dutch_books": [
                {
                    "violation": c.violation,
                    "guaranteed_loss": c.guaranteed_loss,
                    "upfront": c.upfront,
                    "settlement_payoff": c.settlement_payoff,
                    "payoff_by_atom": c.payoff_by_atom,
                    "narrative": c.narrative,
                    "legs": [
                        {"action": "opponent buys" if leg.side > 0 else "opponent sells",
                         "event": leg.event, "price": leg.price}
                        for leg in c.legs
                    ],
                }
                for c in self.certificates if c.is_book()
            ],
        }


def audit_belief(bs: BeliefSet) -> AuditReport:
    """Audit an already-elicited belief set. No model calls."""
    results = checks.run_all(bs)
    certs = [c for c in (dutchbook.build(bs, r) for r in results) if c is not None]
    belief = {k: getattr(bs, k) for k in
              ("a", "b", "not_a", "a_and_b", "a_or_b", "a_given_b", "b_given_a")
              if getattr(bs, k) is not None}
    return AuditReport(context=bs.context, results=results,
                       certificates=certs, belief=belief)


def audit(elicitor: Elicitor, prop_a: str, prop_b: str,
          context: str = "") -> AuditReport:
    """Elicit a belief set for two propositions from a model, then audit it."""
    pa = Proposition(key="A", text=prop_a)
    pb = Proposition(key="B", text=prop_b)
    bs = elicitor.elicit(pa, pb, context)
    return audit_belief(bs)


def render(report: AuditReport) -> str:
    """Human-readable summary for a terminal."""
    lines = []
    status = "COHERENT" if report.coherent else "INCOHERENT"
    lines.append(f"=== Coherence audit: {status} ===")
    if report.context:
        lines.append(f"context: {report.context}")
    lines.append("")
    for r in report.results:
        mark = "ok " if r.coherent else "XX "
        lines.append(f"[{mark}] {r.name:20s} {r.rule}")
        lines.append(f"        {r.detail}  |discrepancy|={r.magnitude:.4f}")
    if report.certificates:
        books = [c for c in report.certificates if c.is_book()]
        if books:
            lines.append("")
            lines.append(f"--- Dutch books found: {len(books)} "
                         f"(total exposure ${report.total_exposure:.4f} per $1 stake) ---")
            for c in books:
                lines.append(f"  * {c.violation}: model loses "
                             f"${c.guaranteed_loss:.4f} regardless of outcome")
                lines.append(f"    {c.narrative}")
                for leg in c.legs:
                    verb = "buys " if leg.side > 0 else "sells"
                    lines.append(f"      opponent {verb} bet on {leg.event} "
                                 f"@ {leg.price:.4f}")
    return "\n".join(lines)
