from C2PA_dataclass import C2PARecord
import c2pa
import json

def detect_c2pa(path: str) -> C2PARecord:
    """
    read C2PA manifest store
    if no C2PA is detected, return has_c2pa=False
    """
    try:
        with c2pa.Context() as ctx:
            with c2pa.Reader(path, context=ctx) as reader:
                manifest_store = json.loads(reader.json())
                active_label = manifest_store.get("active_manifest")
                active_manifest = None

                if active_label:
                    active_manifest = manifest_store.get("manifests", {}).get(active_label)

                return C2PARecord(
                    path=path,
                    has_c2pa=True,
                    active_manifest_label=active_label,
                    active_manifest=active_manifest,
                    manifest_store=manifest_store,
                )

    except Exception as e:
        return C2PARecord(
            path=path,
            has_c2pa=False,
            error=str(e),
        )