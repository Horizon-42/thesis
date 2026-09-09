"""ts_transformer — learned 4D trajectory prediction (vendored iTransformer / PatchTST).

A regular package since 2026-09-09 (package review §4.1). Every module is imported by its
qualified name, ``from ts_transformer.config import TSConfig``; the flat names the modules
used to carry (``config``, ``dataset``, ``train`` — global, and shadowed silently by any
same-named module earlier on ``sys.path``) are gone, and ``tests/test_architecture.py``
refuses one that comes back. The package's parent, ``4dTrajectory/``, is what goes on
``sys.path`` (``__main__.py`` and ``tests/conftest.py`` do it; ``pip install -e
4dTrajectory`` makes it unnecessary, like ``geokit``). ``CLAUDE.md`` is the map.
"""
