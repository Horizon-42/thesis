"""The prior: the upper part of the two-tier model, a language model of the controller (`docs/two_tier/design/prior.md`).

It reads the instruction language (`instructions/`, through the vocabulary's public interface) and the artefact files;
it never imports the executor (`autopilot/`): only the runners join the two (prior design §1,
`tests/test_architecture.py`).
"""
