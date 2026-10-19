"""Prepare recorded examples and verified graph snapshots before training."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from rl_posttraining.core.investigate import arguments_for
from rl_posttraining.integrations import snapshots as snapshot_storage
from rl_posttraining.integrations.files import write_json
from rl_posttraining.integrations.experiment_data import extract


def prepare_dataset(source, destination, manager_args):
    """Join by exact archive path, never just by problem/checkpoint name."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    destination.chmod(0o700)
    state = destination / "environments"
    state.mkdir(mode=0o700)
    catalog = snapshot_storage.index(source, state)
    extraction = extract(source, destination / "pairs")
    if extraction["errors"]:
        raise ValueError("Session extraction failed; inspect pairs/summary.json before training")
    by_archive = {entry["archive"]: identity for identity, entry in catalog.items()}
    grouped = defaultdict(list)
    skipped = []
    for path in sorted((destination / "pairs").glob("*/*/training_data.json")):
        for record in json.loads(path.read_text()):
            archive = str((source / record["metadata"]["archive"]).resolve())
            identity = by_archive.get(archive)
            try:
                arguments_for(record)
                if identity is None:
                    raise ValueError("Source archive has no Neo4j snapshot")
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                skipped.append({"metadata": record["metadata"], "reason": str(error)})
                continue
            grouped[identity].append(record)
    if not grouped:
        raise ValueError("No valid investigate examples have a matching Neo4j snapshot")
    snapshots = []
    for identity, records in sorted(grouped.items()):
        folder, metadata = snapshot_storage.prepare(state, identity, manager_args)
        write_json(folder / "training_data.json", records)
        snapshots.append({"id": identity, "name": metadata["name"],
                          "examples": len(records), "dump_sha256": metadata["dump_sha256"],
                          "schema_sha256": metadata["schema_sha256"]})
    dataset = {"source": str(source), "snapshots": snapshots,
               "examples": sum(s["examples"] for s in snapshots), "skipped": skipped}
    write_json(destination / "dataset.json", dataset)
    print(f"Prepared {dataset['examples']} examples across {len(snapshots)} snapshots; "
          f"skipped {len(skipped)} unmatched/invalid examples.", flush=True)
    return destination


def load_dataset(prepared):
    """Fail before starting containers if prepared artifacts are incomplete."""
    prepared = Path(prepared).resolve()
    dataset = json.loads((prepared / "dataset.json").read_text())
    if not dataset.get("snapshots"):
        raise ValueError("Prepared dataset contains no snapshots")
    for snapshot in dataset["snapshots"]:
        folder, metadata = snapshot_storage.saved_stack(prepared / "environments", snapshot["id"])
        for name in ("backup/neo4j.dump", "project/.artnet/schema.yaml",
                     "project/.artnet/config.yaml", "compose.json", ".env"):
            if not (folder / name).is_file():
                raise ValueError(f"Prepared checkpoint is missing {folder / name}")
        # Reusing prepared data must not silently change the graph or schema.
        for name, checksum in (("backup/neo4j.dump", "dump_sha256"),
                               ("project/.artnet/schema.yaml", "schema_sha256")):
            with (folder / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != snapshot[checksum] or actual != metadata[checksum]:
                raise ValueError(f"Prepared snapshot checksum mismatch: {folder / name}")
        records = json.loads((folder / "training_data.json").read_text())
        if not records or len(records) != snapshot["examples"]:
            raise ValueError(f"Invalid example count for {snapshot['id']}")
        for record in records:
            arguments_for(record)
            archive = (Path(dataset["source"]) / record["metadata"]["archive"]).resolve()
            if str(archive) != metadata["archive"]:
                raise ValueError(f"Example points to the wrong snapshot: {snapshot['id']}")
    return dataset


def load_records(prepared, identity):
    """Read one snapshot's examples after dataset validation has succeeded."""
    folder, _ = snapshot_storage.saved_stack(prepared / "environments", identity)
    return json.loads((folder / "training_data.json").read_text())
