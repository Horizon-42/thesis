"""Third-party forecasting architectures, vendored verbatim.

Each subpackage carries the upstream LICENSE and a PROVENANCE.md pinning the commit,
listing what was copied, and recording every deviation. Nothing in here is repo code:
treat these files as read-only. Adapting a model to this project's data belongs in
``ts_transformer/models.py``, not in a vendored file — an edit here silently breaks the
"byte-identical to upstream" property the provenance notes promise, and with it the
ability to diff against a newer upstream commit.

Why vendored rather than installed: neither upstream is a packaged library, and both
resolve their internal imports through a top-level ``layers/`` directory. Installed side
by side they would collide — both ship a ``layers/Embed.py`` and a
``layers/SelfAttention_Family.py`` with different contents. Package prefixes fix that.
"""
