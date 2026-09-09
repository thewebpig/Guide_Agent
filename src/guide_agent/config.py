"""Small file-loading helpers used by the self-contained demo scene."""

import json
from pathlib import Path


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
