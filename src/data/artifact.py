from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataArtifact:
    name: str
    stage: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)