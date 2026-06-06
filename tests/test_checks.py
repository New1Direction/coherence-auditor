"""Verify the checks fire correctly and that every Dutch book certificate
really does guarantee a loss in every state of the world."""

import math

from coherence_auditor import checks, dutchbook
from coherence_auditor.propositions import BeliefSet


def _approx(a, b, tol=1e-9):
    return abs(a - b) == 0 or abs(a - b) < tol


def test_coherent_set_has_no_violations():
    # A perfectly coherent set: P(A)=0.5, P(B)=0.4, P(A&B)=0.2 (independent),
    # P(AvB)=0.7, P(A|B)=0.5, P(B|A)=0.4, P(~A)=0.5.
    bs = BeliefSet(a=0.5, b=0.4, a_and_b=0.2, a_or_b=0.7,
                   a_given_b=0.5, b_given_a=0.4, not_a=0.5)
    for r in checks.run_all(bs):
        assert r.coherent, f"{r.name} fired on a coherent set: {r.detail}"


def test_negation_overpriced_is_a_dutch_book():
    bs = BeliefSet(a=0.7, not_a=0.6)  # sums to 1.3
    r = checks.negation(bs)
    assert not r.coherent
    assert _approx(r.magnitude, 0.3)
    cert = dutchbook.build(bs, r)
    assert cert.is_book()
    # exactly one of A/~A occurs, so settlement is constant; loss == 0.3
    assert _approx(cert.guaranteed_loss, 0.3)
    # payoff identical across all atoms
    vals = list(cert.payoff_by_atom.values())
    assert max(vals) - min(vals) < 1e-9


def test_negation_underpriced_is_a_dutch_book():
    bs = BeliefSet(a=0.3, not_a=0.2)  # sums to 0.5
    r = checks.negation(bs)
    cert = dutchbook.build(bs, r)
    assert cert.is_book()
    assert _approx(cert.guaranteed_loss, 0.5)


def test_sum_rule_violation_is_a_dutch_book():
    # P(AvB) reported too low: should be 0.5+0.4-0.2=0.7, model says 0.55
    bs = BeliefSet(a=0.5, b=0.4, a_and_b=0.2, a_or_b=0.55)
    r = checks.sum_rule(bs)
    assert not r.coherent
    assert _approx(r.magnitude, 0.15)
    cert = dutchbook.build(bs, r)
    assert cert.is_book()
    assert _approx(cert.guaranteed_loss, 0.15)
    # the defining property: zero net payoff in every atom
    vals = list(cert.payoff_by_atom.values())
    assert all(_approx(v, 0.0) for v in vals), cert.payoff_by_atom


def test_conjunction_fallacy_is_a_dutch_book():
    # Linda: P(A&B)=0.45 but P(A)=0.30 -> fallacy of 0.15
    bs = BeliefSet(a=0.30, b=0.80, a_and_b=0.45)
    r = checks.conjunction_fallacy(bs)
    assert not r.coherent
    assert _approx(r.magnitude, 0.15)
    cert = dutchbook.build(bs, r)
    assert cert.is_book()
    # opponent banks the overshoot upfront and never pays out (settlement >= 0)
    assert cert.upfront > 0
    assert min(cert.payoff_by_atom.values()) >= -1e-9
    assert _approx(cert.guaranteed_loss, 0.15)


def test_product_and_bayes_checks_fire():
    bs = BeliefSet(a=0.5, b=0.4, a_and_b=0.30, a_given_b=0.5, b_given_a=0.4)
    # P(A&B)=0.30 but P(A|B)*P(B)=0.5*0.4=0.20 -> product rule off by 0.10
    pr = checks.product_rule(bs)
    assert _approx(pr.magnitude, 0.10)
    # P(A|B)*P(B)=0.20, P(B|A)*P(A)=0.4*0.5=0.20 -> bayes consistent here
    bc = checks.bayes_consistency(bs)
    assert bc.coherent


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
