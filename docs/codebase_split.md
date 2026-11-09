# Training and reproducibility versus pretrained prediction

Two repositories provide different workflows under the `callmest` account.
The existing [RLBPPred repository](https://github.com/callmest/RLBPPred)
retains training and paper reproduction. The independent
[RLBPPred-inference repository](https://github.com/callmest/RLBPPred-inference)
provides the installable Python package and command, both named `rlbppred`, at
release **v0.2.0**. Training and paper reproduction use this repository's
**v1.1.0** strict401 profile. These are separate software versions, not different
scientific models.

## Codebase responsibilities

| Repository | Responsibility |
|---|---|
| `callmest/RLBPPred` | Strict training reconstruction, frozen split and sampling records, benchmark and candidate validation, downstream analyses, database-export helpers and quantitative figure reproduction |
| `callmest/RLBPPred-inference` | Installation, structure acquisition and validation, Foldseek AA+3Di token generation, fixed SaProt feature extraction, safe pretrained-model loading and prediction |

Training and paper-reproduction commands continue to run from this repository:

```bash
python pipeline/modeling/strict_runtime.py validate
python pipeline/modeling/strict_runtime.py train --help
python scripts/reproduce_figures.py --help
```

Training requires explicitly supplied external assets and `--execute`.
Validating frozen evidence or redrawing panels does not launch training.
Historical model comparisons, original analyses and archived panel data retain
their existing provenance; they are not included in the installed prediction
package. The independent RLBPPredDB application remains a separate local
deployment project.

## Installation and prediction

Install the fixed inference release and retrieve its verified assets:

```bash
python -m pip install "rlbppred[fasta] @ git+https://github.com/callmest/RLBPPred-inference.git@v0.2.0"
rlbppred setup
rlbppred predict-fasta --fasta proteins.fasta --output-dir predictions
```

For development with both checkouts side by side, `python -m pip install
../RLBPPred-inference` also installs the package. Neither method requires a
PyPI release. Once installed, prediction runs independently of the training
checkout; it must not locate code through a sibling-directory import.

Use the inference repository's [v0.2.0 README](https://github.com/callmest/RLBPPred-inference/tree/v0.2.0)
for the accepted FASTA-to-prediction
commands, required optional dependencies and recorded acceptance limits.
Ordinary FASTA is not passed directly to the MLP: sequence-matched structures,
Foldseek AA+3Di tokens and the fixed SaProt encoder are required first.
The package must not silently substitute sequence-only embeddings or a
different structure/encoder model.

## Large assets: official encoder and GitHub model release

The SaProt encoder is downloaded by the user from the authors' official
[westlake-repl/SaProt_650M_AF2 repository](https://huggingface.co/westlake-repl/SaProt_650M_AF2),
not repackaged in either code repository. The fixed revision is
`1b8cddd7bad34f6f5627422b4908309840c78fef`:

```bash
curl --fail --location \
  https://huggingface.co/westlake-repl/SaProt_650M_AF2/resolve/1b8cddd7bad34f6f5627422b4908309840c78fef/SaProt_650M_AF2.pt \
  --output SaProt_650M_AF2.pt
```

Expected size: 2,606,464,143 bytes. Expected SHA256:

```text
bc110567bc75511c7f46eb3e7d109fd6e81051baad471e22c5306ed71653cd52
```

The project's strict MLP bundle (`model.json` and `model.safetensors`) is supplied
as separate assets in the
[inference v0.2.0 Release](https://github.com/callmest/RLBPPred-inference/releases/tag/v0.2.0).
The fixed download URLs are:

- [model.json](https://github.com/callmest/RLBPPred-inference/releases/download/v0.2.0/model.json)
- [model.safetensors](https://github.com/callmest/RLBPPred-inference/releases/download/v0.2.0/model.safetensors)

Use only the release filenames and checksums recorded there and in the
inference package's pinned asset manifest. An absent release asset is
not replaced by an unverified mirror. Large weights are excluded from Git and
the code wheel. No Zenodo upload, archival DOI or PyPI publication is required
by this distribution arrangement. Third-party encoder and structure terms
continue to apply.

## Model and input boundary

The scientific model is the frozen strict SaProt-650M-AF2 plus 300-member MLP
ensemble. The prediction interface consumes raw, unscaled, 1,280-dimensional
embeddings and applies the fitted training scaler itself. Do not provide
already standardized embeddings or representations from a different encoder.

The installed interface accepts a `safetensors` model bundle rather than the
original pickle-capable checkpoint. Preparing the bundle converts the trusted
frozen `strict_ensemble.pt` only after checking its known SHA256:

```text
72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41
```

Conversion is not training and preserves the fitted scaler, all 300 members'
parameters and inference arithmetic. The safe weight file's SHA256 is:

```text
e9f2db20805d6212606851a29e69b8d286ba443b8ae0e2f22872060e963ac3b5
```

## Compatibility and verification

`pipeline/modeling/strict_runtime.py predict` remains the compatibility and
numerical-regression reference. Its trusted legacy-checkpoint interface is
distinct from the safe-bundle interface. Existing `validate`, `screen` and
`train` entrypoints and the frozen scientific evidence remain in this repository.

Software acceptance distinguishes synthetic input-contract tests, regression
using actual pretrained weights and full-proteome biological-data agreement.
None authorizes retraining or replacement of frozen predictions. The inference
repository's validation record identifies which checks actually ran; installation
alone does not establish end-to-end acceptance or general biological accuracy.
