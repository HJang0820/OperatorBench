# Common data schema v0.1

This is a versioned data interface for physical fields and operator-learning
tasks. Its current implementation covers fixed shared Cartesian grids and a
Darcy adapter. It is an engineering interface, not a claim of a unique or
previously unpublished data representation.

## Dataset record

| Section | Required meaning |
| --- | --- |
| `schema_version` | Common-interface version, currently `"0.1"` |
| `problem` | PDE name, equation, and spatial domain |
| `sample_ids` | Unique identifiers for rows of the supplied dataset |
| `fields` | Named physical arrays, quantity, units, components, and field-specific time coordinates |
| `coordinates` | Shared spatial points, original grid shape, `ij` indexing, and C-order flattening |
| `task` | Named `inputs`, `targets`, and `fixed_fields` |
| `conditions` | Boundary definition, initial-condition field references, and fixed parameters |
| `splits` | Disjoint train/validation/test sample IDs |
| `provenance` | Reference-generation method and the source dataset's available metadata |

Every field has axes `(sample, time, point, component)`. A static field has
`time=None` and one storage slot; that slot does not mean an observation at
physical time zero. A time-dependent field has its own strictly increasing
time-coordinate array. `unit=None` explicitly means that units are unspecified.
Floating-point values retain the source dtype and physical scale.

## Darcy examples

| Task | Inputs | Targets | Fixed field | Boundary |
| --- | --- | --- | --- | --- |
| `a_to_u` | `a` | `u` | `f` | Zero Dirichlet |
| `f_to_u` | `f` | `u` | `a` | Zero Dirichlet |

For 1,400 samples on an 85 x 85 grid, each of `a`, `f`, and `u` has shape
`(1400, 1, 7225, 1)`. Spatial coordinates have shape `(7225, 2)` and
`grid_shape=(85, 85)`. Splits contain 1,000/200/200 sample IDs.
The adapter checks positivity of `a` and the declared zero boundary of `u`.
It also verifies that each declared fixed field is identical across samples.

The adapter maps the existing `A/F/U` arrays to `a/f/u`; it does not solve the
PDE or normalize data. Reordering/reshaping is reversible. Source files and
existing training/checkpoint formats are preserved.

## Terminal usage

From the repository root, convert a current split NPZ with embedded metadata:

```bash
python -m scripts.check_common_schema --dataset data/darcy/source.npz --output data/darcy/common_v01.npz --dataset-id darcy_experiment_01 --report outputs/schema_check.json
```

Replace the input path with your existing split dataset. The output must be a
new path. The command verifies array equality after saving/loading and checks
that the source file hash is unchanged.

Six-array notebook caches without embedded coordinates, source fields, or
metadata require an explicit `--legacy-config` JSON file. The example
`configs/legacy_a_to_u.json` describes one historical experiment only; verify
its assumptions against your cache before using it. Supplied metadata is
labelled as an external declaration, not recovered evidence of how the cache
was generated.

```python
from operatorbench.data.schema import load_operator_dataset, select_split

dataset = load_operator_dataset("data/darcy/common_v01.npz")
train = select_split(dataset, "train")
input_name = dataset["task"]["inputs"][0]
input_values = train[input_name]
target_values = train[dataset["task"]["targets"][0]]
```

Fit normalization on the selected training values only. Model-specific input
construction remains a separate operation. The current training/evaluation
commands continue to use the existing Darcy split NPZ. They reuse the common
fixed-field check to compare the stored values with the declared constant,
using the train/validation splits for training and the test split for evaluation.
The full common schema is validated when converting or loading a common NPZ.

## Checks and limits

```bash
python -m unittest discover -s tests -p test_common_schema.py -v
```

Checks cover rectangular-grid ordering, exact persistence, split overlap,
fixed-field consistency, invalid values/shapes, and per-field time coordinates.
A synthetic temporal layout check is not a Burgers solver or experiment.
Sample IDs identify source rows, not physical cases shared across resolutions.
Moving meshes, unstructured grids, interpolation, and additional boundary
solvers are outside v0.1. Recording a condition does not implement its solver.

On 2026-11-06, local conversions of the existing Darcy `a_to_u` and Gaussian
sine-source `f_to_u` datasets (1,000/200/200; 85 x 85) preserved all available
source field arrays exactly and left source hashes unchanged. This is a data
interface check, not a new numerical-accuracy or model-performance result.
