"""Dutch book construction.

A coherence violation is not just an aesthetic flaw: by de Finetti's theorem it
means an opponent can offer the model a set of bets, each priced at the model's
own stated probability (so each looks fair *to the model*), whose combined payoff
is a guaranteed loss in every possible state of the world.

We use the standard bet convention:

    A "unit bet on event E at price p" costs p to enter and pays 1 if E occurs,
    0 otherwise. The model, holding P(E) = p, considers p a fair price and is
    willing to take *either* side at that price.

A *certificate* is a list of bet legs (sign, event, price) plus the proof that
the opponent comes out ahead in every atom of the world. For the negation and
sum-rule books that proof is a *constant* net payoff across all four atoms while
the net cost is nonzero; for the conjunction book it is a *dominated* payoff
(non-negative in every atom) with cash banked upfront. `guaranteed_loss` is the
resulting gap — money the model forfeits no matter what happens. The `_assemble`
helper handles both: constant-payoff books settle on that constant, non-constant
books are scored on their worst atom (`min(payoffs)`).

The four atoms of the A,B world (in fixed order) are:
    TT = A and B,  TF = A and not B,  FT = not A and B,  FF = neither.
Indicator payoff vectors over (TT, TF, FT, FF):
    A    -> (1,1,0,0)
    B    -> (1,0,1,0)
    A&B  -> (1,0,0,0)
    AvB  -> (1,1,1,0)
    ~A   -> (0,0,1,1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import CheckResult
from .propositions import BeliefSet

ATOMS = ("A&B", "A&~B", "~A&B", "~A&~B")

_INDICATOR = {
    "A":   (1.0, 1.0, 0.0, 0.0),
    "B":   (1.0, 0.0, 1.0, 0.0),
    "A&B": (1.0, 0.0, 0.0, 0.0),
    "AvB": (1.0, 1.0, 1.0, 0.0),
    "~A":  (0.0, 0.0, 1.0, 1.0),
}


@dataclass
class BetLeg:
    """One bet in the portfolio. side=+1 the opponent buys this bet from the
    model (model is short the event); side=-1 the opponent sells it (model long).
    We frame everything from the *opponent's* cashflow so a positive guaranteed
    profit for the opponent is the model's guaranteed loss."""

    side: int          # +1 or -1
    event: str         # key into _INDICATOR
    price: float       # the model's stated probability for `event`

    def net_payoff_vector(self) -> tuple[float, ...]:
        ind = _INDICATOR[self.event]
        # opponent buying the bet (side +1): pays price now, receives indicator
        # later -> payoff to opponent is +indicator, cost is +price.
        return tuple(self.side * x for x in ind)

    def opponent_upfront(self) -> float:
        # buying costs the opponent `price`; selling pays the opponent `price`.
        return -self.side * self.price


@dataclass
class Certificate:
    violation: str
    legs: list[BetLeg]
    guaranteed_loss: float          # to the model; > 0 means a real Dutch book
    settlement_payoff: float        # opponent's net payoff, same in every atom
    upfront: float                  # opponent's net cash before settlement
    narrative: str = ""
    payoff_by_atom: dict = field(default_factory=dict)

    def is_book(self, tol: float = 1e-9) -> bool:
        return self.guaranteed_loss > tol


def _assemble(violation: str, legs: list[BetLeg], narrative: str) -> Certificate:
    # Net payoff must be identical across all four atoms for this to be a true
    # book; we record it and assert constancy up to float noise.
    vecs = [leg.net_payoff_vector() for leg in legs]
    by_atom = {}
    for i, atom in enumerate(ATOMS):
        by_atom[atom] = sum(v[i] for v in vecs)
    payoffs = list(by_atom.values())
    settlement = payoffs[0]
    spread = max(payoffs) - min(payoffs)
    upfront = sum(leg.opponent_upfront() for leg in legs)
    # Opponent's guaranteed profit = worst-case total over atoms (settlement is
    # constant when spread ~ 0) plus upfront cash.
    if spread > 1e-9:
        # not a constant-payoff portfolio; not a clean book
        guaranteed = upfront + min(payoffs)
    else:
        guaranteed = upfront + settlement
    return Certificate(
        violation=violation,
        legs=legs,
        guaranteed_loss=round(guaranteed, 12),
        settlement_payoff=round(settlement, 12),
        upfront=round(upfront, 12),
        narrative=narrative,
        payoff_by_atom={k: round(v, 12) for k, v in by_atom.items()},
    )


def from_negation(bs: BeliefSet, r: CheckResult) -> Certificate:
    """p + q != 1 where p=P(A), q=P(~A). Exactly one of A,~A occurs, so a bet on
    each pays exactly 1 in total. If p+q>1 the opponent sells both (model
    overpays p+q for a sure 1); if p+q<1 the opponent buys both."""
    p, q = bs.a, bs.not_a
    overpriced = (p + q) > 1.0
    # If overpriced (p+q>1): opponent SELLS both bets to the model (side -1);
    # opponent receives p+q now, pays out exactly 1 (one of A,~A always occurs).
    # If underpriced: opponent BUYS both (side +1).
    side = -1 if overpriced else +1
    legs = [BetLeg(side=side, event="A", price=p),
            BetLeg(side=side, event="~A", price=q)]
    note = ("model overprices the partition" if overpriced
            else "model underprices the partition")
    return _assemble("negation", legs,
                     f"{note}: P(A)+P(~A)={p+q:.4f} != 1")


def from_sum_rule(bs: BeliefSet, r: CheckResult) -> Certificate:
    """Portfolio (A, B, -A&B, -AvB) has identically zero payoff in every atom,
    so its only cost is the price discrepancy d = P(A)+P(B)-P(A&B)-P(AvB).
    The opponent takes whichever side makes d work against the model."""
    # discrepancy in the *check* is P(AvB) - [P(A)+P(B)-P(A&B)].
    # Define d = P(A)+P(B)-P(A&B)-P(AvB) = -r.discrepancy.
    d = -(r.discrepancy)
    # The zero-payoff portfolio from the model's side is: long A, long B,
    # short A&B, short AvB, costing d. If d>0 the model is paying d for nothing,
    # so the opponent simply lets the model hold it (opponent takes the mirror,
    # collecting d). If d<0 the opponent holds it themselves.
    if d > 0:
        # opponent is the counterparty to the model's losing portfolio:
        # opponent short A, short B, long A&B, long AvB -> receives d upfront,
        # zero net settlement.
        legs = [BetLeg(-1, "A", bs.a), BetLeg(-1, "B", bs.b),
                BetLeg(+1, "A&B", bs.a_and_b), BetLeg(+1, "AvB", bs.a_or_b)]
    else:
        legs = [BetLeg(+1, "A", bs.a), BetLeg(+1, "B", bs.b),
                BetLeg(-1, "A&B", bs.a_and_b), BetLeg(-1, "AvB", bs.a_or_b)]
    return _assemble("sum_rule", legs,
                     f"zero-payoff portfolio mispriced by |d|={abs(d):.4f}")


def from_conjunction_fallacy(bs: BeliefSet, r: CheckResult) -> Certificate:
    """P(A&B) > P(conjunct). A&B implies that conjunct, so the conjunct's bet
    pays >= the conjunction's bet in every atom. Opponent buys the (cheaper-
    should-be) conjunct... actually buys the conjunct and sells the overpriced
    conjunction, pocketing the overshoot now and never paying out."""
    # pick the conjunct that is exceeded
    conj_key, conj_val = ("A", bs.a) if (bs.a is not None and bs.a <= (bs.b if bs.b is not None else 1.0)) else ("B", bs.b)
    # opponent BUYS conjunct (side +1), SELLS A&B (side -1)
    legs = [BetLeg(+1, conj_key, conj_val), BetLeg(-1, "A&B", bs.a_and_b)]
    return _assemble("conjunction_fallacy", legs,
                     f"P(A&B)={bs.a_and_b:.4f} > P({conj_key})={conj_val:.4f}; "
                     f"opponent banks {bs.a_and_b - conj_val:.4f} upfront with "
                     "non-negative settlement")


_BUILDERS = {
    "negation": from_negation,
    "sum_rule": from_sum_rule,
    "conjunction_fallacy": from_conjunction_fallacy,
}


def build(bs: BeliefSet, r: CheckResult) -> Certificate | None:
    """Construct a Dutch book certificate for a violated check, if we have a
    builder for it. Returns None for checks we don't (yet) monetize directly
    (product_rule, bayes_consistency reduce to combinations of the above)."""
    if r.coherent:
        return None
    builder = _BUILDERS.get(r.name)
    if builder is None:
        return None
    return builder(bs, r)
