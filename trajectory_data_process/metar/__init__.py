"""Surface wind observations (METAR/ASOS) for the observed baseline's speed gate.

The harvest keeps tracks as broadcast and evaluation joins the wind at READ time
(``evaluation.wind``); nothing here rewrites a track, an arrival roster or a record.
"""
