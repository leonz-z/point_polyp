from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Cfg:
    data: dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        if item not in self.data:
            raise AttributeError(item)
        return self.data[item]


def load_cfg(path: str | Path) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Cfg(data)
