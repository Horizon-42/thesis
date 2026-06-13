from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
AERODYNAMIC_MODEL_DIR = REPO_ROOT / "aerodynamic_model"
TRAJECTORY_OPTIMIZATION_DIR = REPO_ROOT / "4dTrajectory" / "optimization"


def install_import_paths() -> None:
    for path in (AERODYNAMIC_MODEL_DIR, TRAJECTORY_OPTIMIZATION_DIR):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


install_import_paths()
