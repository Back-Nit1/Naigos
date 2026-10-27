"""Versioned, framework-neutral trajectory datasets.

This package deliberately depends on NumPy and the Python standard library,
not JAX.  Collection code may hand JAX arrays to :class:`TrajectoryWriter`
because NumPy's array protocol converts them at the boundary, while validation
and reading remain usable in a small offline-analysis environment.
"""

from .io import (
    DatasetIOError,
    ImmutableDatasetError,
    IntegrityError,
    PartialDatasetError,
    TrajectoryReader,
    TrajectoryWriter,
    ValidationReport,
    load_metadata,
    validate_chunk_arrays,
    validate_dataset,
)
from .schema import (
    ACTION_COMPONENTS,
    FORMAT_NAME,
    FORMAT_VERSION,
    TERMINATION_CAUSES,
    DatasetMetadata,
    FieldSpec,
    PolicyIdentity,
    Provenance,
    SchemaError,
    TrajectorySchema,
    UnsupportedVersionError,
    canonical_json_bytes,
    default_trajectory_schema,
)

__all__ = [
    "ACTION_COMPONENTS",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "TERMINATION_CAUSES",
    "DatasetIOError",
    "DatasetMetadata",
    "FieldSpec",
    "ImmutableDatasetError",
    "IntegrityError",
    "PartialDatasetError",
    "PolicyIdentity",
    "Provenance",
    "SchemaError",
    "TrajectoryReader",
    "TrajectorySchema",
    "TrajectoryWriter",
    "UnsupportedVersionError",
    "ValidationReport",
    "canonical_json_bytes",
    "default_trajectory_schema",
    "load_metadata",
    "validate_chunk_arrays",
    "validate_dataset",
]
