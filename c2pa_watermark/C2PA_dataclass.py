import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class C2PARecord:
    path: str
    has_c2pa: bool
    active_manifest_label: Optional[str] = None
    active_manifest: Optional[Dict[str, Any]] = None
    manifest_store: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)