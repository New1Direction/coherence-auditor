"""MCP server exposing the coherence auditor as tools any agent can call.

Run (stdio transport):
    pip install "mcp[cli]"
    python -m coherence_auditor.mcp_server
or register the module with your MCP client.

Two tools:

  audit_coherence(probabilities, context)
      Audit a belief set the *caller* already produced. `probabilities` is a
      dict with any of: a, b, not_a, a_and_b, a_or_b, a_given_b, b_given_a.
      No model is called. This is the "an agent checks its own intermediate
      beliefs before acting" path.

  audit_propositions(prop_a, prop_b, context)
      Elicit the full belief set for two propositions from a Claude model, then
      audit it. Requires ANTHROPIC_API_KEY in the server's environment.

Both return the AuditReport.to_dict() payload, including any Dutch book
certificates and the total exposure.
"""

from __future__ import annotations

from .audit import audit, audit_belief
from .elicit import belief_from_dict

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "mcp[cli]" to run the MCP server') from e

mcp = FastMCP("coherence-auditor")


@mcp.tool()
def audit_coherence(probabilities: dict, context: str = "") -> dict:
    """Check a set of probabilities for violations of the sum/product rules and
    return any Dutch books (guaranteed-loss bet portfolios) they imply.

    probabilities: dict with any subset of keys
        a, b, not_a, a_and_b, a_or_b, a_given_b, b_given_a
        where a=P(A), b=P(B), not_a=P(~A), a_and_b=P(A&B), a_or_b=P(A v B),
        a_given_b=P(A|B), b_given_a=P(B|A). Each value in [0, 1].
    context: optional free-text describing what was being conditioned on.

    Returns a report: {coherent, total_exposure, checks[], dutch_books[]}.
    total_exposure is the guaranteed loss (per $1 staked) an opponent can
    extract given these prices; 0 means coherent.
    """
    bs = belief_from_dict({**probabilities, "context": context})
    return audit_belief(bs).to_dict()


@mcp.tool()
def audit_propositions(prop_a: str, prop_b: str, context: str = "",
                       model: str = "claude-3-5-haiku-latest") -> dict:
    """Elicit probabilities for two propositions from a Claude model, then audit
    them for coherence. Requires ANTHROPIC_API_KEY in the environment.

    prop_a, prop_b: natural-language propositions (A and B).
    context: optional conditioning context shown to the model.
    model: Anthropic model id used for elicitation.
    """
    from .elicit import AnthropicElicitor
    elicitor = AnthropicElicitor(model=model)
    return audit(elicitor, prop_a, prop_b, context).to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
