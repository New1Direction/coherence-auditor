"""Data model for propositions and the probabilities a model assigns them.

A *belief set* is the collection of probabilities an LLM has stated over a small
family of related propositions (A, B, their conjunction, disjunction, negation,
and the relevant conditionals). The coherence checks operate on belief sets;
they never call a model themselves. That separation is deliberate: elicitation
is noisy and model-specific, but the logic the document derives from lattice
symmetries is not, so we keep the math pure and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Proposition:
    """A single proposition the model can be asked about.

    `key` is a short stable identifier ("A", "B", ...). `text` is the natural
    language the model actually sees during elicitation.
    """

    key: str
    text: str


@dataclass
class BeliefSet:
    """Probabilities a model assigned over a related family of propositions.

    Every field is the model's stated probability for one quantity, in [0, 1].
    Fields left as None are simply not checked; each coherence check declares
    which fields it needs and is skipped when they are absent. This lets you run
    a partial audit (e.g. just the conjunction fallacy) without eliciting the
    full set.

    The conditioning context C from the document is implicit: it is whatever was
    in the prompt when these numbers were elicited. We record it as `context`
    for provenance but it does not enter the arithmetic, exactly because the
    sum/product rules must hold *for any fixed C*.
    """

    a: Optional[float] = None          # P(A | C)
    b: Optional[float] = None          # P(B | C)
    not_a: Optional[float] = None      # P(~A | C)
    a_and_b: Optional[float] = None    # P(A & B | C)
    a_or_b: Optional[float] = None     # P(A | B  ... no: P(A v B | C)
    a_given_b: Optional[float] = None  # P(A | B, C)
    b_given_a: Optional[float] = None  # P(B | A, C)

    context: str = ""
    prop_a: Optional[Proposition] = None
    prop_b: Optional[Proposition] = None
    raw: dict = field(default_factory=dict)  # anything the elicitor wants to stash

    def __post_init__(self) -> None:
        for name in (
            "a", "b", "not_a", "a_and_b", "a_or_b", "a_given_b", "b_given_a",
        ):
            v = getattr(self, name)
            if v is not None and not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{name}={v} is not a probability in [0, 1]. "
                    "A model that returns this is already incoherent, but the "
                    "elicitor should clamp/flag it before it reaches here."
                )
