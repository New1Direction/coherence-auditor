# coherence-auditor

Probe an LLM's stated probabilities for violations of the **sum and product
rules**, and turn every violation into a **Dutch book** — an explicit set of
bets, each priced at the model's *own* stated probability, that guarantees the
model loses money no matter what happens.

## Why

The lattice-symmetry derivation of probability (Cox / Jaynes / Knuth) shows the
sum and product rules are not axioms you choose but *necessary* consequences of
classical logic:

```
P(A v B | C) = P(A | C) + P(B | C) - P(A & B | C)        # sum rule
P(A & B | C) = P(A | B, C) * P(B | C)                    # product rule
```

An LLM is often described as computing `P(next token | context)` — a
conditional plausibility, which is exactly the Bayesian object. But the token
softmax is coherent by construction; it sums to one. What is *not* guaranteed
coherent is the number the model **states** when you actually ask it `P(A)`.
Those elicited judgments are a behavioral output, not a distribution read off
the logits, and they routinely violate the sum and product rules — the same way
human judgments do (the conjunction fallacy is the classic case). This tool
measures that gap in a model's *stated* probabilities and, crucially,
**prices it**: by de Finetti's theorem an incoherent set of probabilities is
equivalent to accepting a guaranteed-loss bet, so each violation is reported as
a dollar figure of guaranteed loss per $1 staked.

That framing — a *guaranteed-loss certificate* per belief set — is sharper than
a calibration plot: it's adversarial and constructive. It tells you not just
that the model is miscalibrated but exactly which bets extract money from it.

## What it checks

| check | rule | book? |
|---|---|---|
| `negation` | `P(A) + P(~A) = 1` | yes |
| `sum_rule` | `P(A v B) = P(A) + P(B) - P(A & B)` | yes |
| `product_rule` | `P(A & B) = P(A\|B) P(B)` | reduces to others |
| `bayes_consistency` | `P(A\|B) P(B) = P(B\|A) P(A)` | reduces to others |
| `conjunction_fallacy` | `P(A & B) <= min(P(A), P(B))` | yes |

Each Dutch book certificate proves the portfolio extracts money in **every
atom** of the A,B world. The `negation` and `sum_rule` books are *constant-payoff*
portfolios (identical settlement in all four atoms) that cost a nonzero amount;
the `conjunction_fallacy` book is a *dominance* book (non-negative settlement
everywhere, with a positive amount banked upfront). Either way the gap is the
guaranteed loss.

A set that violates **only** `product_rule` or `bayes_consistency` is reported as
incoherent but with `$0` priced exposure: those reduce to the others and are not
monetized as standalone certificates, so the check fires without its own book.

## Install

```bash
pip install -e .                 # core, no deps
pip install -e ".[anthropic]"    # to elicit from a live Claude model
pip install -e ".[mcp]"          # to run the MCP server
```

## Use it as a library

```python
from coherence_auditor import BeliefSet, audit_belief, render

bs = BeliefSet(a=0.30, b=0.85, a_and_b=0.55, a_or_b=0.90, not_a=0.75)
report = audit_belief(bs)
print(render(report))
print(report.total_exposure)     # 0.60  -> $0.60 extractable per $1
report.to_dict()                 # full structured payload
```

## Use it against a live model

```python
from coherence_auditor import AnthropicElicitor, audit, render
# needs ANTHROPIC_API_KEY
report = audit(
    AnthropicElicitor(model="claude-3-5-haiku-latest"),
    prop_a="Linda is a bank teller",
    prop_b="Linda is active in the feminist movement",
    context="Linda is 31, single, outspoken, and majored in philosophy.",
)
print(render(report))
```

Each probability is elicited in its own call so the model can't anchor one
answer on another and hide the incoherence.

## Use it as an MCP tool

```bash
python -m coherence_auditor.mcp_server
```

Exposes two tools:

- `audit_coherence(probabilities, context)` — audit a belief set the caller
  already produced. No model call. This is the "an agent checks its own
  intermediate beliefs before committing to an action" path.
- `audit_propositions(prop_a, prop_b, context, model)` — elicit from a Claude
  model, then audit. Needs `ANTHROPIC_API_KEY`.

## Demo

```bash
python examples/linda.py
```

## Layout

```
coherence_auditor/
  propositions.py   # BeliefSet data model
  checks.py         # the five coherence checks (pure math, no model)
  dutchbook.py      # violation -> guaranteed-loss certificate
  elicit.py         # Elicitor interface, MockElicitor, AnthropicElicitor
  audit.py          # orchestration + reporting
  mcp_server.py     # MCP server
tests/test_checks.py
examples/linda.py
```

The math (`checks.py`, `dutchbook.py`) never touches a model and is fully
unit-tested, so swapping in an OpenAI/local elicitor is one method.

## License

MIT
