from __future__ import annotations
from dataclasses import dataclass


@dataclass 
class Node:
    """ Decision Tree Node """
    is_leaf = bool = False
    prediction = int | None
    feature_index = int | None
    threshold: float | None 
    left: Node | None
    right: Node | None

    

    