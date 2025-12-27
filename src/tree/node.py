from __future__ import annotations
from dataclasses import dataclass


@dataclass 
class Node:
    """ Decision Tree Node """
    is_leaf = bool = False
    prediction = int | None
    feature_index = int | None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    

    