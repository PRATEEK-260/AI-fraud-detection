"""Shared data structures — every agent emits Case objects, nothing else."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional
import uuid

Decision = Literal["allow", "escalate", "block"]


@dataclass
class Evidence:
    signal: str          # e.g. "velocity_spike", "shared_device_id"
    value: str            # human-readable value for the case file
    weight: float         # contribution to confidence, 0-1


@dataclass
class Case:
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    entity_id: str = ""              # user_id / card_id / device_id / dispute_id
    entity_type: str = ""            # "transaction" | "account" | "document" | "dispute"
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0          # 0-1
    cost_estimate: float = 0.0       # expected cost if wrongly blocked
    decision: Decision = "allow"
    reasoning_text: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
