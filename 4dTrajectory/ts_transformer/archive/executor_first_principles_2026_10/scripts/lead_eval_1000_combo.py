#!/usr/bin/env python
"""`lead_eval_1000` for the combined arms only: angle and speed leads together (heading 4 s)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead_eval_1000 as run   # noqa: E402

run.arms = lambda: [("combo_a4_s4", {"heading": 4.0, "angle": 4.0, "speed": 4.0}, None),
                    ("base", dict(run.BASE), None)]
raise SystemExit(run.main())
