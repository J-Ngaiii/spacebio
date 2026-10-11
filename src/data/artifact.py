from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

@dataclass
class DataArtifact:
    name: str
    stage: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata[self.stage] = {
            'reader_dict': {
                'filenames': [],
                'paths': []
            },
            'writer_dict': {
                'filenames': [],
                'paths': []
            }
        }

    