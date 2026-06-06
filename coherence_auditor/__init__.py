"""coherence-auditor: probe stated probabilities for sum/product-rule violations
and price each as a Dutch book (a guaranteed-loss bet portfolio).

The public surface mirrors the README: a pure-math core (`checks`, `dutchbook`)
that never calls a model, plus elicitation and orchestration helpers.
"""

from __future__ import annotations

from . import checks, dutchbook
from .audit import AuditReport, audit, audit_belief, render
from .checks import ALL_CHECKS, CheckResult, run_all
from .dutchbook import BetLeg, Certificate, build
from .elicit import (
    AnthropicElicitor,
    Elicitor,
    MockElicitor,
    belief_from_dict,
)
from .propositions import BeliefSet, Proposition

__all__ = [
    "ALL_CHECKS",
    "AnthropicElicitor",
    "AuditReport",
    "BeliefSet",
    "BetLeg",
    "Certificate",
    "CheckResult",
    "Elicitor",
    "MockElicitor",
    "Proposition",
    "audit",
    "audit_belief",
    "belief_from_dict",
    "build",
    "checks",
    "dutchbook",
    "render",
    "run_all",
]
