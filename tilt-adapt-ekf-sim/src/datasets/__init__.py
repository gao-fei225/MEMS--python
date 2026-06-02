# Dataset management module

from .schema import (
    create_empty_dataset,
    get_n_samples,
    flatten_dict,
    unflatten_dict,
    SCHEMA_VERSION,
    REQUIRED_FIELDS,
)
from .validate import (
    validate_dataset,
    validate_dataset_soft,
    print_dataset_info,
    DatasetValidationError,
    ensure_quaternion_continuity,
)
from .serialize import (
    save_npz,
    load_npz,
    dataset_to_dict,
)
from .repoimu import (
    RepoIMUTStickTrial,
    iter_repoimu_tstick_paths,
    load_repoimu_tstick_trial,
)

__all__ = [
    "create_empty_dataset",
    "get_n_samples",
    "flatten_dict",
    "unflatten_dict",
    "SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "validate_dataset",
    "validate_dataset_soft",
    "print_dataset_info",
    "DatasetValidationError",
    "ensure_quaternion_continuity",
    "save_npz",
    "load_npz",
    "dataset_to_dict",
    "RepoIMUTStickTrial",
    "iter_repoimu_tstick_paths",
    "load_repoimu_tstick_trial",
]
