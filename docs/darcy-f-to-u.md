# Darcy source-to-solution example

The task is `f -> u` for `-div(a grad u) = f` on the unit square, with fixed
`a = 1` and homogeneous Dirichlet boundaries. Each sample has a different
source. The generator reuses `build_dataset` and `solve_darcy_fd` from the
coefficient-input task.

## Configuration

`configs/darcy_data_f_to_u.json` defines the data path, grid, splits, seeds,
fixed coefficient, and source distribution. The defaults are 85 x 85 nodes
and 1,000/200/200 samples with split seeds 2030/2031/2032.

The Gaussian sine series uses K=4, amplitude=1 and smoothness=2:

```text
f_i(x,y) = amplitude * sum[p,q=1..K]
           xi_i,pq * (p^2+q^2)^(-smoothness/2)
           * sin(pi*p*x) * sin(pi*q*y),   xi_i,pq ~ Normal(0,1)
```

These are explicit experiment settings, not a claim of a universal standard
distribution. Reference fields are computed using the existing flux-form
five-point FDM with arithmetic face averages. Computation uses float64 and
stored A/F/U fields use float32.

The matching training configurations are:

| Configuration | Model | Branch activation | Initial learning rate |
| --- | --- | --- | --- |
| `configs/darcy_f_to_u_deeponet.json` | Fully-connected DeepONet | GELU in branch hidden layers | 1e-4 |
| `configs/darcy_f_to_u_fno.json` | FNO | Not applicable | 1e-3 |

The DeepONet trunk retains Tanh in its hidden and output layers. Both examples
use 25,000 optimizer updates, batch size 20, and StepLR every 5,000 updates
with gamma 0.5. The learning rates differ between models. The original
`configs/darcy_baseline.json` retains the Tanh `a -> u` experiment.

## Terminal sequence

Run from the repository root. Data generation loads an existing file only
when its metadata and arrays match the requested configuration.

```bash
python -m scripts.generate_darcy_splits --config configs/darcy_data_f_to_u.json
python -u -m scripts.train_darcy --config configs/darcy_f_to_u_deeponet.json --model deeponet --device cpu --run-name f_to_u_deeponet_01
python -u -m scripts.train_darcy --config configs/darcy_f_to_u_fno.json --model fno --device cpu --run-name f_to_u_fno_01
python -u -m scripts.evaluate_darcy --device cpu --deeponet-checkpoint outputs/f_to_u_deeponet_01/best.pt --fno-checkpoint outputs/f_to_u_fno_01/best.pt --output-dir outputs/f_to_u_evaluation_01 --sample-indices 0 5 19
```

Use `--device cuda` for training and evaluation with a compatible GPU
environment. Resume by adding `--resume` to the original training command,
keeping its configuration and run name. Use new run names for new experiments.

Normalization is fitted on `F_train` and `U_train` only and saved in each
checkpoint. Validation/test reuse those statistics. Evaluation verifies the
task, data hash, metadata, coordinates, normalization and inference source
hashes before predicting. Reference solutions and predictions are compared
on their physical scale. All test samples contribute to metrics even when
only selected examples are plotted.

To export a completed evaluation:

```bash
python -m scripts.export_darcy_report --result-dir outputs/f_to_u_evaluation_01 --report-name f_to_u_evaluation_01
```

Export uses the actual test count and selected figure indices. Large datasets
and checkpoints remain excluded from Git. Existing checkpoints require the
source revision recorded by their experiment; this port preserves that
compatibility check and does not rewrite historical checkpoints.

## Quick checks

```bash
python -m unittest discover -s tests -p test_darcy_workflow.py -v
python -m unittest discover -s tests -p test_common_schema.py -v
```

The workflow check generates small temporary datasets, trains both models for
two updates on CPU, checks checkpoint resume, evaluates all held-out samples,
and checks user-selected figures. It exercises both `a -> u` and `f -> u`.
It is a functional integration check, not a benchmark accuracy experiment.

## Grid and task boundaries

Change `N` and the output dataset path in the data configuration, then update
the training grid, dataset path and split settings consistently. Explicit
sensor counts and FNO mode counts must fit the selected grid. Generating a new
grid does not demonstrate resolution generalization of an existing checkpoint.

This workflow keeps the existing Darcy split NPZ for training. The common
schema conversion and persistence interface remains available separately.
