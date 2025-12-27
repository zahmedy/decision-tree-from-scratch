from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Node:
    is_leaf: bool = False
    prediction: int | None = None
    feature_index: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None
