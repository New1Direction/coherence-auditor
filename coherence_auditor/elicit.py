"""Elicitors turn a pair of propositions into a BeliefSet by asking a model.

The interface is one method, `elicit(prop_a, prop_b, context) -> BeliefSet`.
Two implementations ship here:

  * MockElicitor   - deterministic, no network; used in tests and demos. You can
                     seed it with deliberate incoherence to exercise the books.
  * AnthropicElicitor - asks a real Claude model for each probability with a
                     prompt that forces a bare number, then clamps/parses.

Elicitation is the noisy, model-specific part of the system. Everything that
matters mathematically lives in checks.py / dutchbook.py and never touches a
model, so you can swap in an OpenAI/local elicitor by implementing this one
method.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from .propositions import BeliefSet, Proposition


class Elicitor(Protocol):
    def elicit(self, prop_a: Proposition, prop_b: Proposition,
               context: str = "") -> BeliefSet: ...


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
class MockElicitor:
    """Returns a fixed BeliefSet. Handy for tests and for demoing a known
    incoherence without spending tokens."""

    def __init__(self, belief: BeliefSet):
        self._belief = belief

    def elicit(self, prop_a, prop_b, context="") -> BeliefSet:
        bs = self._belief
        bs.prop_a, bs.prop_b, bs.context = prop_a, prop_b, context
        return bs


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])")

_QUESTIONS = {
    "a":         "P({A})",
    "b":         "P({B})",
    "not_a":     "P(NOT {A})",
    "a_and_b":   "P({A} AND {B})",
    "a_or_b":    "P({A} OR {B})",
    "a_given_b": "P({A} GIVEN that {B})",
    "b_given_a": "P({B} GIVEN that {A})",
}

_SYS = (
    "You are estimating probabilities. For each question reply with ONLY a "
    "single number between 0 and 1 (inclusive), no words, no percent sign. "
    "Treat each question independently and answer honestly with your best "
    "estimate given the context."
)


class AnthropicElicitor:
    """Elicits each probability in its own call so the model can't anchor one
    answer on another (which would mask incoherence). Requires the `anthropic`
    package and ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str = "claude-3-5-haiku-latest",
                 fields: tuple[str, ...] = ("a", "b", "not_a", "a_and_b",
                                            "a_or_b", "a_given_b", "b_given_a")):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install anthropic, and set ANTHROPIC_API_KEY"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.fields = fields

    def _ask(self, question: str, context: str) -> float:
        ctx = f"Context: {context}\n\n" if context else ""
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=16,
            system=_SYS,
            messages=[{"role": "user", "content": f"{ctx}Estimate: {question}"}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = _NUM.search(text.strip())
        if not m:
            raise ValueError(f"could not parse a probability from: {text!r}")
        val = float(m.group(1))
        return min(1.0, max(0.0, val))

    def elicit(self, prop_a, prop_b, context="") -> BeliefSet:
        subs = {"A": prop_a.text, "B": prop_b.text}
        values, raw = {}, {}
        for field in self.fields:
            q = _QUESTIONS[field].format(**subs)
            v = self._ask(q, context)
            values[field] = v
            raw[field] = {"question": q, "value": v}
        bs = BeliefSet(**values, context=context, prop_a=prop_a, prop_b=prop_b)
        bs.raw = raw
        return bs


def belief_from_dict(d: dict) -> BeliefSet:
    """Build a BeliefSet from a plain dict of field->probability. Lets callers
    (e.g. the MCP tool) pass probabilities they elicited elsewhere."""
    allowed = {"a", "b", "not_a", "a_and_b", "a_or_b", "a_given_b", "b_given_a"}
    fields = {k: float(v) for k, v in d.items() if k in allowed and v is not None}
    return BeliefSet(**fields, context=d.get("context", ""))
