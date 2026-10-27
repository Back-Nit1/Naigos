"""Versioned, chunked trajectory data is framework-neutral and fail-closed."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from naigos.trajectory import (
    ACTION_COMPONENTS,
    DatasetMetadata,
    FieldSpec,
    ImmutableDatasetError,
    IntegrityError,
    PartialDatasetError,
    PolicyIdentity,
    Provenance,
    SchemaError,
    TrajectoryReader,
    TrajectorySchema,
    TrajectoryWriter,
    UnsupportedVersionError,
    canonical_json_bytes,
    default_trajectory_schema,
    validate_chunk_arrays,
    validate_dataset,
)


def _schema():
    return default_trajectory_schema(
        n_agents=2,
        ego_features=10,
        threat_slots=3,
        threat_features=12,
        friend_slots=1,
        friend_features=7,
        extra_dimensions={"global_feature": 5},
        extra_fields=(
            FieldSpec(
                "privileged/global_state",
                "float32",
                ("global_feature",),
                "1",
                "Optional centralized training state unavailable to the actor.",
                {"method": "componentwise", "components": [
                    {"name": f"state_{i}", "source_unit": "1", "transform": "identity"}
                    for i in range(5)
                ]},
                tuple(f"state_{i}" for i in range(5)),
            ),
            FieldSpec(
                "resource/thermal_margin",
                "float32",
                ("agent",),
                "K",
                "Example future resource-state extension.",
                {"method": "identity"},
            ),
        ),
    )


def _metadata(schema=None):
    return DatasetMetadata(
        dataset_id="baseline-heldout-s999",
        created_at="2026-10-26T12:00:00Z",
        description="Unit-test shard; no measured research result.",
        schema=schema or _schema(),
        provenance=Provenance(
            theatre="owens_valley",
            route="west_east",
            scenario="heldout_v1",
            seed=999,
            code_commit="a" * 40,
            code_dirty=False,
            environment_configuration={
                "n_blue": 2,
                "action_names": list(ACTION_COMPONENTS),
                "dt_s": 2.0,
            },
            policy=PolicyIdentity(
                policy_id="theatre_1000",
                kind="checkpoint",
                framework="jax-flax-optax",
                artifact_sha256="b" * 64,
                configuration={"actor": "deepsets_attention"},
            ),
        ),
    )


def _chunk(schema, start: int, records: int):
    arrays = {}
    for field in schema.fields:
        shape = (records, *schema.resolved_shape(field))
        arrays[field.name] = np.zeros(shape, dtype=field.dtype)

    arrays["episode/id"][:] = np.uint64(17)
    arrays["step/index"][:] = np.arange(start, start + records, dtype=np.uint32)
    arrays["status/reset"][:] = False
    if start == 0:
        arrays["status/reset"][0] = True
    arrays["status/alive"][:] = True
    arrays["status/reached"][:] = False
    arrays["status/termination_cause"][:] = np.uint8(0)
    arrays["observation/threat_mask"][:, :, :2] = True
    arrays["observation/friend_mask"][:] = True

    requested = arrays["action/requested"]
    requested[..., 0] = 0.10
    requested[..., 1] = -0.20
    requested[..., 2] = 0.60
    executed = arrays["action/executed"]
    executed[:] = requested
    executed[..., 0] += 0.05
    arrays["cbf/feasible"][:] = True
    arrays["cbf/intervention_l2"][:] = np.linalg.norm(executed - requested, axis=-1)

    fuel = 100.0 - np.arange(start, start + records, dtype=np.float32)[:, None]
    arrays["resource/fuel_before"][:] = fuel
    arrays["resource/fuel_after"][:] = fuel - 0.5
    arrays["reward/fuel_used"][:] = 0.5
    arrays["resource/thermal_margin"][:] = 20.0
    arrays["privileged/global_state"][:] = np.arange(5, dtype=np.float32)
    return arrays


def _episode_chunk(schema, episode: int, start: int, records: int):
    arrays = _chunk(schema, start, records)
    arrays["episode/id"][:] = np.uint64(episode)
    return arrays


def test_chunked_round_trip_preserves_arrays_metadata_and_content_hash(tmp_path):
    root = tmp_path / "trajectory"
    metadata = _metadata()
    first = _chunk(metadata.schema, 0, 2)
    second = _chunk(metadata.schema, 2, 3)

    with TrajectoryWriter(root, metadata) as writer:
        d0 = writer.append(first)
        d1 = writer.append(second)
        assert d0["records"] == 2 and d1["records"] == 3
        # The writer retains only indices/metadata, never campaign arrays.
        assert not any(isinstance(value, np.ndarray) for value in vars(writer).values())

    report = validate_dataset(root)
    assert report.complete
    assert (report.records, report.chunks) == (5, 2)
    assert len(report.content_sha256) == 64
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["content_sha256"] == report.content_sha256
    assert manifest["metadata"]["sha256"] == json.loads(
        (root / "metadata.commit.json").read_text()
    )["sha256"]

    reader = TrajectoryReader(root)
    assert reader.metadata == metadata
    chunks = list(reader.iter_chunks())
    assert len(chunks) == 2
    for name in first:
        np.testing.assert_array_equal(chunks[0][name], first[name])
        np.testing.assert_array_equal(chunks[1][name], second[name])
    all_rows = reader.read_all()
    assert all_rows["episode/id"].shape == (5,)
    np.testing.assert_array_equal(all_rows["step/index"], np.arange(5, dtype=np.uint32))

    with pytest.raises(ImmutableDatasetError, match="existing path"):
        TrajectoryWriter(root, metadata)
    with pytest.raises(ImmutableDatasetError, match="finalized"):
        TrajectoryWriter.resume(root)


def test_corrupt_chunk_is_rejected_before_deserialization(tmp_path):
    root = tmp_path / "trajectory"
    metadata = _metadata()
    with TrajectoryWriter(root, metadata) as writer:
        writer.append(_chunk(metadata.schema, 0, 2))

    payload = root / "chunks" / "00000000.npz"
    damaged = bytearray(payload.read_bytes())
    damaged[len(damaged) // 2] ^= 0x01
    payload.write_bytes(damaged)

    with pytest.raises(IntegrityError, match="sha256 mismatch"):
        validate_dataset(root)
    with pytest.raises(IntegrityError, match="sha256 mismatch"):
        list(TrajectoryReader(root).iter_chunks())


def test_metadata_corruption_is_detected_even_while_partial(tmp_path):
    root = tmp_path / "trajectory"
    writer = TrajectoryWriter(root, _metadata())
    writer.append(_chunk(writer.metadata.schema, 0, 1))
    path = root / "metadata.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="byte count mismatch"):
        validate_dataset(root, allow_partial=True)


def test_unknown_format_and_schema_versions_fail_closed():
    document = _metadata().to_dict()
    future_format = copy.deepcopy(document)
    future_format["format_version"] = "2.0.0"
    with pytest.raises(UnsupportedVersionError, match="format version"):
        DatasetMetadata.from_dict(future_format)

    future_schema = copy.deepcopy(document)
    future_schema["schema"]["version"] = "1.1.0"
    with pytest.raises(UnsupportedVersionError, match="schema version"):
        DatasetMetadata.from_dict(future_schema)


def test_on_disk_unknown_version_is_reported_as_a_migration_issue(tmp_path):
    root = tmp_path / "future"
    root.mkdir()
    (root / "chunks").mkdir()
    document = _metadata().to_dict()
    document["format_version"] = "2.0.0"
    payload = canonical_json_bytes(document)
    (root / "metadata.json").write_bytes(payload)
    (root / "metadata.commit.json").write_bytes(canonical_json_bytes({
        "file": "metadata.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }))
    with pytest.raises(UnsupportedVersionError, match="format version"):
        validate_dataset(root, allow_partial=True)


def test_partial_dataset_is_explicit_readable_by_opt_in_and_resumable(tmp_path):
    root = tmp_path / "trajectory"
    metadata = _metadata()
    writer = TrajectoryWriter(root, metadata)
    writer.append(_chunk(metadata.schema, 0, 2))

    assert not (root / "manifest.json").exists()
    with pytest.raises(PartialDatasetError, match="incomplete"):
        validate_dataset(root)
    with pytest.raises(PartialDatasetError):
        TrajectoryReader(root)

    partial = validate_dataset(root, allow_partial=True)
    assert not partial.complete
    assert (partial.records, partial.chunks) == (2, 1)
    assert partial.content_sha256 is None
    assert partial.partial_reasons == ("manifest.json is absent",)
    assert len(list(TrajectoryReader(root, allow_partial=True).iter_chunks())) == 1

    resumed = TrajectoryWriter.resume(root, expected_metadata=metadata)
    resumed.append(_chunk(metadata.schema, 2, 1))
    final = resumed.finalize()
    assert final.complete and final.records == 3


def test_sequence_order_is_checked_within_and_across_chunks(tmp_path):
    schema = _schema()
    root = tmp_path / "ordered"
    writer = TrajectoryWriter(root, _metadata(schema))
    writer.append(_episode_chunk(schema, 17, 0, 2))
    writer.append(_episode_chunk(schema, 17, 2, 1))

    skipped = _episode_chunk(schema, 17, 4, 1)
    with pytest.raises(SchemaError, match=r"expected step/index=3, found 4"):
        writer.append(skipped)

    regressed = _episode_chunk(schema, 17, 0, 2)
    regressed["status/reset"][:] = False
    regressed["step/index"][:] = np.array([3, 2], dtype=np.uint32)
    with pytest.raises(SchemaError, match=r"expected step/index=4, found 2"):
        # The first row is the valid next step; the second proves ordering is
        # checked inside the supplied chunk before any bytes are committed.
        writer.append(regressed)

    assert not (root / "chunks" / "00000002.npz").exists()


def test_new_episode_requires_reset_and_closed_episode_cannot_resume(tmp_path):
    schema = _schema()
    root = tmp_path / "episodes"
    writer = TrajectoryWriter(root, _metadata(schema))
    writer.append(_episode_chunk(schema, 17, 0, 2))

    missing_reset = _episode_chunk(schema, 18, 0, 1)
    missing_reset["status/reset"][:] = False
    with pytest.raises(SchemaError, match="must start at step/index=0 with status/reset=true"):
        writer.append(missing_reset)

    writer.append(_episode_chunk(schema, 18, 0, 1))
    resumed = TrajectoryWriter.resume(root, expected_metadata=_metadata(schema))
    with pytest.raises(SchemaError, match="episode 17 resumes after a different episode"):
        resumed.append(_episode_chunk(schema, 17, 0, 1))


def test_terminal_cause_is_one_shot_per_agent_across_chunks(tmp_path):
    schema = _schema()
    root = tmp_path / "terminal"
    writer = TrajectoryWriter(root, _metadata(schema))

    first = _episode_chunk(schema, 17, 0, 1)
    first["status/termination_cause"][0, 0] = np.uint8(6)  # timeout
    writer.append(first)

    repeated = _episode_chunk(schema, 17, 1, 1)
    repeated["status/termination_cause"][0, 0] = np.uint8(6)
    with pytest.raises(SchemaError, match="agent 0 has a repeated terminal cause"):
        writer.append(repeated)


def test_dataset_validator_rejects_sequence_corruption_with_valid_file_hashes(tmp_path):
    schema = _schema()
    root = tmp_path / "tampered-sequence"
    writer = TrajectoryWriter(root, _metadata(schema))
    writer.append(_episode_chunk(schema, 17, 0, 1))
    writer.append(_episode_chunk(schema, 17, 1, 1))

    payload_path = root / "chunks" / "00000001.npz"
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["step/index"][:] = np.uint32(9)
    with payload_path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)

    descriptor_path = root / "chunks" / "00000001.json"
    descriptor = json.loads(descriptor_path.read_text())
    payload = payload_path.read_bytes()
    descriptor["bytes"] = len(payload)
    descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
    descriptor_path.write_bytes(canonical_json_bytes(descriptor))

    with pytest.raises(IntegrityError, match=r"expected step/index=1, found 9"):
        validate_dataset(root, allow_partial=True)
    with pytest.raises(IntegrityError, match=r"expected step/index=1, found 9"):
        list(TrajectoryReader(root, allow_partial=True).iter_chunks())
    with pytest.raises(IntegrityError, match=r"expected step/index=1, found 9"):
        TrajectoryWriter.resume(root)


def test_orphaned_payload_is_partial_not_silently_ignored(tmp_path):
    root = tmp_path / "trajectory"
    metadata = _metadata()
    writer = TrajectoryWriter(root, metadata)
    writer.append(_chunk(metadata.schema, 0, 1))
    (root / "chunks" / "00000001.npz").write_bytes(b"interrupted")

    report = validate_dataset(root, allow_partial=True)
    assert any("uncommitted chunk payloads" in reason for reason in report.partial_reasons)
    with pytest.raises(PartialDatasetError, match="reviewed"):
        TrajectoryWriter.resume(root)
    with pytest.raises(PartialDatasetError, match="partial artifacts"):
        writer.finalize()


def test_schema_enforces_action_contract_dtypes_and_intervention():
    schema = _schema()
    arrays = _chunk(schema, 0, 1)
    assert schema.dimension_map["action"] == 3
    assert schema.field_map["action/requested"].components == ACTION_COMPONENTS

    wrong_dtype = dict(arrays)
    wrong_dtype["action/requested"] = arrays["action/requested"].astype(np.float64)
    with pytest.raises(SchemaError, match="expected dtype float32"):
        validate_chunk_arrays(schema, wrong_dtype)

    wrong_intervention = dict(arrays)
    wrong_intervention["cbf/intervention_l2"] = np.zeros_like(arrays["cbf/intervention_l2"])
    with pytest.raises(SchemaError, match="intervention_l2"):
        validate_chunk_arrays(schema, wrong_intervention)

    simultaneous = _chunk(schema, 0, 1)
    simultaneous["reward/shotdown"][:] = 1.0
    simultaneous["reward/terrain_violation"][:] = 1.0
    simultaneous["constraint_cost/aircraft_loss"][:] = 1.0
    simultaneous["status/alive"][:] = False
    simultaneous["status/termination_cause"][:] = np.uint8(3)  # terrain takes precedence
    validate_chunk_arrays(schema, simultaneous)
    simultaneous["status/termination_cause"][:] = np.uint8(2)
    with pytest.raises(SchemaError, match="precedence"):
        validate_chunk_arrays(schema, simultaneous)

    document = schema.to_dict()
    document["dimensions"]["action"] = 4
    for field in document["fields"]:
        if field["name"].startswith("action/"):
            field["components"].append("fourth_axis")
    with pytest.raises(SchemaError, match="exactly 3"):
        TrajectorySchema.from_dict(document)


def test_schema_and_validation_modules_do_not_import_jax():
    import naigos.trajectory.io as trajectory_io
    import naigos.trajectory.schema as trajectory_schema

    for module in (trajectory_io, trajectory_schema):
        source = Path(module.__file__).read_text()
        assert "import jax" not in source
        assert "from jax" not in source
