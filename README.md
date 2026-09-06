# Entropic Flow Network for Bayesian Network Structure Learning

This repository contains experimental code related to the paper **Entropic Flow Network for Bayesian Network Structure Learning**. The paper proposes the Entropic Flow Network (EFN), which introduces structural entropy into GFlowNet to jointly account for data faithfulness, structural uncertainty, and directed acyclicity. It also uses a window balance loss to improve the global search for candidate Bayesian network structures.

> **Implementation note:** The complete experimental environment described in the paper uses PyTorch 1.13 and Python 3.8. The current repository snapshot is implemented with **JAX, Haiku, and Optax**. This README therefore distinguishes the method described in the paper from the code that is currently available. The accompanying `requirements.txt` follows the actual imports in this repository.

## Method Overview

Bayesian network structure learning aims to recover a directed acyclic graph (DAG) that best represents the dependencies among variables in a dataset. EFN starts from an empty graph and progressively generates candidate DAGs through two types of actions: adding a directed edge or terminating the construction process. It learns a generative policy that samples graphs with probability proportional to their rewards.

The EFN framework described in the paper consists of two connected components:

1. **Structure uncertainty reduction**
   - Uses BDeu or linear-Gaussian likelihood to measure data faithfulness.
   - Uses K-dimensional structural entropy over an encoding tree to measure uncertainty within a state.
   - Uses a symmetric divergence between the in-degree distributions of consecutive states to measure structural changes across states.
   - Uses incremental cycle detection and action masking to ensure that every generated graph is a DAG.
2. **Candidate structure optimization**
   - Uses a Transformer to represent the current graph and parameterize the edge-addition and termination policies.
   - Uses forward and backward policies to satisfy GFlowNet flow consistency.
   - Uses a window balance loss to regulate structural entropy changes along a trajectory and a policy entropy regularizer to preserve exploration.

The unified reward in the paper can be summarized as:

```text
R(s) = R_A(s) * [(1 - lambda_S) * R_D(s) + lambda_S * R_S(s)]
```

Here, `R_A`, `R_D`, and `R_S` denote the acyclicity constraint, data-faithfulness reward, and structural-uncertainty reward, respectively.

## Experiments Reported in the Paper

The paper reports results on seven benchmark Bayesian networks from the bnlearn repository:

| Dataset | Nodes | Edges | Parameters | Domain | Code directory |
| --- | ---: | ---: | ---: | --- | --- |
| Asia | 8 | 8 | 18 | Medical diagnosis | `small-datasets` |
| Cancer | 5 | 4 | 10 | Medical diagnosis | `small-datasets` |
| Earthquake | 5 | 4 | 10 | Risk assessment | `small-datasets` |
| Sachs | 11 | 17 | 178 | Cellular signaling | `small-datasets` |
| Survey | 6 | 6 | 21 | Social sciences | `small-datasets` |
| Alarm | 37 | 46 | 509 | Medical monitoring | `mid` |
| Water | 32 | 66 | 10083 | Water treatment | `mid` |

F1 and ROC-AUC are used to evaluate structural recovery, while MAE and MSE are used to evaluate probabilistic inference. The comparison methods are PC, HC, H2PC, and DAG-GFlowNet. The paper reports the following EFN configuration:

| Item | Paper setting |
| --- | --- |
| Runtime | Python 3.8, PyTorch 1.13, NVIDIA RTX 4090 with 24 GB memory |
| Transformer | 4 layers, 8 attention heads, hidden dimension 256 |
| Optimizer | Adam |
| Learning rate / batch size | `1e-4` / `32` |
| Reward weight | `lambda_S = 0.8` |
| Window weight | `lambda_W = 0.3` |
| Smoothing constant | `epsilon = 1e-6` |
| Encoding-tree height / window size | `K = 2` / `zeta = 5` |
| Repeated runs | 5 |

For the inference evaluation, the paper generates 100 random evidence sets for each network, conditions on 30% of the variables, and estimates the marginal probabilities of the remaining variables using 10,000 likelihood-weighting samples.

## Repository Structure

```text
code_icdm/
|-- README.md
|-- requirements.txt
|-- small-datasets/                 # Small networks and the basic DAG-GFlowNet workflow
|   |-- train.py                    # Training, posterior sampling, and structural metrics
|   |-- combine_training.py         # Sequential multi-dataset workflow
|   |-- continuous_generating.py    # Builds pgmpy models from posterior.npy
|   `-- dag_gflownet/
|       |-- env.py                  # DAG environment and acyclic action masking
|       |-- gflownet.py             # Basic GFlowNet implementation
|       |-- nets/                   # Attention and Transformer networks
|       |-- scores/                 # BDe, BGe, and BIC scores
|       `-- utils/                  # Data, sampling, metrics, and replay buffer
|-- mid/                            # Medium networks and entropy-enhanced experiments
|   |-- train.py                    # Training, scheduling, checkpoints, and recovery
|   |-- continuous_training.py      # Sequential runner for six medium datasets
|   |-- combine_training.py         # Combined training, generation, and inference workflow
|   |-- continuous_generating.py    # Generates pgmpy models from posterior samples
|   `-- dag_gflownet/
|       |-- env_v4.py               # Entropy-enhanced DAG environment used by train.py
|       |-- gflownet_c_b.py         # GFlowNet implementation used by train.py
|       |-- nets/                   # Jraph and Transformer graph representation
|       |-- scores/                 # BDe, BGe, and BIC scores
|       `-- utils/                  # Schedulers, recovery, metrics, and checkpoints
`-- 1/                              # Historical documentation; not used at runtime
```

## Installation

The repository contains cached files produced by CPython 3.9 and 3.10. **Python 3.10** is therefore recommended. The following example uses Windows PowerShell:

```powershell
Set-Location C:\Users\asus02\Downloads\code_icdm
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default dependency file installs the standard CPU build of JAX. GPU-enabled JAX installations depend on the CUDA and cuDNN versions available on the target machine. Install a matching `jax`/`jaxlib` build by following the official JAX installation instructions, then install the remaining dependencies.

## Quick Start

The scripts use relative paths. Run them from the corresponding code directory.

### Small Datasets from the Paper

The following command trains the Asia network and writes the results to `small-datasets/output/asia`:

```powershell
Set-Location C:\Users\asus02\Downloads\code_icdm\small-datasets
New-Item -ItemType Directory -Force output | Out-Null
python train.py --batch_size 32 --num_iterations 20000 --output_folder output/asia asia
```

Replace the final positional argument with `cancer`, `earthquake`, `sachs`, or `survey` to train another small network from the paper. Global options must appear before the dataset subcommand.

A synthetic linear-Gaussian dataset can be used for a shorter test:

```powershell
python train.py --batch_size 32 --num_iterations 1000 --output_folder output/er5 erdos_renyi_lingauss --num_variables 5 --num_edges 5 --num_samples 100
```

### Medium Datasets from the Paper

Alarm and Water correspond to the following subcommands in the current code:

- `alarm_interventional_bic`
- `water_interventional_bic`

The `mid` directory also supports `barley_interventional_bic`, `child_interventional_bic`, `insurance_interventional_bic`, and `mildew_interventional_bic`. These additional networks are not included in the main results table of the paper.

The intended command format for `mid/train.py` is:

```powershell
Set-Location C:\Users\asus02\Downloads\code_icdm\mid
New-Item -ItemType Directory -Force output | Out-Null
python train.py --lr 1e-4 --lr_scheduler reduce_on_plateau --output_folder output/alarm_interventional_bic alarm_interventional_bic
```

However, the current `mid/train.py` snapshot registers the `child_interventional_bic` subcommand twice. The program therefore exits while initializing its argument parser. One of the duplicate registrations must be removed before running a medium-network experiment. No source code was modified as part of this documentation task.

### Generate Bayesian Networks from Posterior Samples

The training scripts produce `posterior.npy`. Each `continuous_generating.py` script reads the first five posterior adjacency matrices, estimates the conditional probability tables using maximum likelihood, and saves `models/model_0.pkl` through `models/model_4.pkl`.

After training all small datasets:

```powershell
Set-Location C:\Users\asus02\Downloads\code_icdm\small-datasets\output
python ..\continuous_generating.py
```

After training all medium datasets, run `..\continuous_generating.py` from `mid/output` in the same way.

## Common Training Options

| Option | `small-datasets` default | `mid` CLI default | Description |
| --- | ---: | ---: | --- |
| `--num_envs` | 8 | 8 | Number of parallel DAG environments |
| `--lr` | `1e-5` | `1e-3` | Initial Adam learning rate |
| `--batch_size` | 32 | 32 | Replay-buffer sample size |
| `--num_iterations` | 20,000 | 100,000 | Number of training iterations |
| `--replay_capacity` | 100,000 | 100,000 | Replay-buffer capacity |
| `--prefill` | 1,000 | 1,000 | Random-policy prefill iterations |
| `--num_samples_posterior` | 1,000 | 1,000 | Number of posterior samples |
| `--num_workers` | 4 | 4 | Number of scoring worker processes |
| `--seed` | 0 | 0 | Random seed |
| `--output_folder` | `output` | `output` | Output directory |

The `mid` entry point additionally exposes learning-rate scheduling, checkpointing, training-state recovery, and contrastive-learning options. Run `python train.py --help` to view the complete argument list after resolving the duplicate subcommand described above.

## Data and Outputs

Benchmark network definitions are obtained through `pgmpy.utils.get_example_model`. If `data/<name>.txt` is missing, the code generates samples with `BayesianModelSampling.forward_sample` and caches them in the local `data/` directory. The first run of the Sachs continuous or interventional dataset requires internet access to download the source data.

A complete training run normally produces:

| File | Description |
| --- | --- |
| `arguments.json` | Arguments used for the run |
| `data.csv` | Training data |
| `graph.pkl` | Ground-truth benchmark graph |
| `model.npz` | Haiku/JAX model parameters |
| `replay_buffer.npz` | Experience replay buffer |
| `posterior.npy` | Posterior DAG adjacency-matrix samples |
| `results.json` | Training time, expected SHD, F1, ROC-AUC, and related structural metrics |
| `records.jsonl` / `abcd.jsonl` | Compact records for small / medium experiments |

The `mid` workflow can also save the best model, periodic checkpoints, and complete training states.

## Relationship Between the Paper and This Code Snapshot

The current repository implements sequential DAG generation, acyclic action masking, Bayesian scores, Transformer policies, flow-balance training, posterior sampling, and structural metrics. The following differences are important when interpreting or reproducing results:

- The paper reports a PyTorch 1.13 implementation, while this repository snapshot uses JAX.
- The paper uses K-dimensional encoding-tree entropy, divergence between consecutive states, and a window balance loss of length `zeta`. The current `mid/env_v4.py` uses a simplified in-degree-distribution entropy and a short-window score combination; it is not the complete implementation of Algorithm A2 from the paper appendix.
- The structural-entropy combination path in `small-datasets/env.py` is currently commented out, so that directory is closer to the basic DAG-GFlowNet baseline.
- The `config` object at the end of `mid/train.py` fixes the effective training length and batch size at 60,000 and 258. The corresponding command-line options currently do not change those two effective values, which differs from the paper's batch size of 32.
- Contrastive-loss computation is commented out in the active `mid` training entry point, and the logged `contrastive_loss` is set to zero. Its command-line options are not equivalent to the paper's `lambda_S` or `lambda_W`.
- Reproducing the paper's MAE and MSE results requires downstream probabilistic-inference evaluation. Both `combine_training.py` files reference `inference_*.py` files that are not present in this repository, so the current snapshot cannot directly reproduce the MAE/MSE table. Use `train.py` and `continuous_generating.py` for structure learning and model generation.
- Although the `mid` CLI lists several graph priors, its current factory function only implements the `uniform` prior.

The numerical results in the paper should therefore be treated as reported paper results rather than results that can be reproduced from this snapshot without additional work. Exact paper reproduction requires the reported PyTorch EFN implementation, the complete structural-entropy and window-balance algorithms, and the missing probabilistic-inference scripts.

## References and Citation

The baseline DAG-GFlowNet code is derived from the [JAX DAG-GFlowNet implementation](https://github.com/tristandeleu/jax-dag-gflownet) by Tristan Deleu and collaborators. Benchmark networks are obtained from the [bnlearn Bayesian Network Repository](https://www.bnlearn.com/bnrepository/).

The supplied manuscript is an anonymous UAI 2026 review submission and does not contain author information. Add the authors, publication information, and a permanent URL when they become available. Until then, the following temporary citation can be used:

```bibtex
@unpublished{anonymous2026efn,
  title = {Entropic Flow Network for Bayesian Network Structure Learning},
  note = {Submitted to the 42nd Conference on Uncertainty in Artificial Intelligence (UAI 2026), review manuscript},
  year = {2026}
}
```
