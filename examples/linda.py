"""The classic conjunction fallacy ("Linda the bank teller"), run through the
auditor with a mock elicitor seeded with the incoherent numbers a model
typically gives. Run:  python examples/linda.py

Swap MockElicitor for AnthropicElicitor (and set ANTHROPIC_API_KEY) to test a
live model instead of the canned numbers.
"""

from coherence_auditor import BeliefSet, MockElicitor, audit, render

# A = "Linda is a bank teller"
# B = "Linda is active in the feminist movement"
# The fallacy: people (and models) rate P(A & B) above P(A).
seeded = BeliefSet(
    a=0.30,       # P(bank teller)
    b=0.85,       # P(feminist)
    a_and_b=0.55, # P(bank teller AND feminist)  -- impossibly high vs P(A)
    a_or_b=0.90,
    a_given_b=0.30,
    b_given_a=0.80,
    not_a=0.75,   # P(not bank teller) -- note 0.30 + 0.75 = 1.05, also incoherent
)

elicitor = MockElicitor(seeded)
report = audit(
    elicitor,
    prop_a="Linda is a bank teller",
    prop_b="Linda is active in the feminist movement",
    context="Linda is 31, single, outspoken, and majored in philosophy.",
)
print(render(report))
print()
print(f"total exposure: ${report.total_exposure:.4f} per $1 staked")
