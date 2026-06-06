"""Coherence checks.

Each check corresponds to a rule the lecture derives as *necessary* from the
symmetries of classical logic. An LLM that truly embodied P(proposition | context)
would satisfy all of them identically. They don't, so each check reports a signed
discrepancy and a magnitude; `dutchbook.py` turns a nonzero magnitude into a
guaranteed-loss certificate.

A check is skipped (returns None) when the belief set lacks the fields it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .propositions import BeliefSet


@dataclass
class CheckResult:
    name: str
    rule: str                 # the identity that should hold
    discrepancy: float        # signed: lhs - rhs (0.0 == coherent)
    magnitude: float          # abs(discrepancy)
    detail: str               # human-readable
    tolerance: float = 1e-9

    @property
    def coherent(self) -> bool:
        return self.magnitude <= self.tolerance


def negation(bs: BeliefSet) -> Optional[CheckResult]:
    """P(A) + P(~A) = 1.  The simplest possible incoherence."""
    if bs.a is None or bs.not_a is None:
        return None
    d = (bs.a + bs.not_a) - 1.0
    return CheckResult(
        name="negation",
        rule="P(A) + P(~A) = 1",
        discrepancy=d,
        magnitude=abs(d),
        detail=f"P(A)={bs.a:.4f} + P(~A)={bs.not_a:.4f} = {bs.a + bs.not_a:.4f}",
    )


def sum_rule(bs: BeliefSet) -> Optional[CheckResult]:
    """P(A v B) = P(A) + P(B) - P(A & B)."""
    if None in (bs.a, bs.b, bs.a_and_b, bs.a_or_b):
        return None
    lhs = bs.a_or_b
    rhs = bs.a + bs.b - bs.a_and_b
    d = lhs - rhs
    return CheckResult(
        name="sum_rule",
        rule="P(A v B) = P(A) + P(B) - P(A & B)",
        discrepancy=d,
        magnitude=abs(d),
        detail=f"P(AvB)={lhs:.4f} vs P(A)+P(B)-P(A&B)={rhs:.4f}",
    )


def product_rule(bs: BeliefSet) -> Optional[CheckResult]:
    """P(A & B) = P(A | B) * P(B)."""
    if None in (bs.a_and_b, bs.a_given_b, bs.b):
        return None
    lhs = bs.a_and_b
    rhs = bs.a_given_b * bs.b
    d = lhs - rhs
    return CheckResult(
        name="product_rule",
        rule="P(A & B) = P(A | B) * P(B)",
        discrepancy=d,
        magnitude=abs(d),
        detail=f"P(A&B)={lhs:.4f} vs P(A|B)*P(B)={rhs:.4f}",
    )


def bayes_consistency(bs: BeliefSet) -> Optional[CheckResult]:
    """P(A | B) * P(B) = P(B | A) * P(A).

    Both sides equal P(A & B), so they must agree. This is the commutativity
    the document calls out as the source of Bayes' theorem.
    """
    if None in (bs.a_given_b, bs.b, bs.b_given_a, bs.a):
        return None
    lhs = bs.a_given_b * bs.b
    rhs = bs.b_given_a * bs.a
    d = lhs - rhs
    return CheckResult(
        name="bayes_consistency",
        rule="P(A|B)*P(B) = P(B|A)*P(A)",
        discrepancy=d,
        magnitude=abs(d),
        detail=f"P(A|B)*P(B)={lhs:.4f} vs P(B|A)*P(A)={rhs:.4f}",
    )


def conjunction_fallacy(bs: BeliefSet) -> Optional[CheckResult]:
    """P(A & B) must not exceed P(A) or P(B).

    The Linda problem. Unlike the others this is an *inequality*; discrepancy is
    the amount of the overshoot (positive == fallacy present, <=0 == fine).
    """
    if bs.a_and_b is None or (bs.a is None and bs.b is None):
        return None
    bounds = [v for v in (bs.a, bs.b) if v is not None]
    worst = bs.a_and_b - min(bounds)  # >0 means conjunction beats a conjunct
    return CheckResult(
        name="conjunction_fallacy",
        rule="P(A & B) <= min(P(A), P(B))",
        discrepancy=worst,
        magnitude=max(worst, 0.0),
        detail=f"P(A&B)={bs.a_and_b:.4f} vs min conjunct={min(bounds):.4f}",
    )


ALL_CHECKS: list[Callable[[BeliefSet], Optional[CheckResult]]] = [
    negation,
    sum_rule,
    product_rule,
    bayes_consistency,
    conjunction_fallacy,
]


def run_all(bs: BeliefSet) -> list[CheckResult]:
    out = []
    for chk in ALL_CHECKS:
        r = chk(bs)
        if r is not None:
            out.append(r)
    return out
