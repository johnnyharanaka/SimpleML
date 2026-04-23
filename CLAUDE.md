# SimpleML

Modular, registry-based ML training framework. Components are registered by name and composed via YAML config or fluent Python `API`.

## Commands

```bash
# Install
uv pip install -e ".[dev]"

# Train
python -m simpleml config.yaml
python -m simpleml config.yaml --resume checkpoints/last.pt

# Test
uv run pytest

# Lint / Format
uv run ruff check .
uv run ruff format .
```

## Architecture

All components live in `simpleml/` and are registered via `simpleml/registry.py`. Config shape is always `{name: str, params: dict}`.

| Module | Contents |
|---|---|
| `models/` | TimmModel |
| `losses/` | CrossEntropy, BCE, Focal, NTXent, SupCon, Triplet |
| `metrics/` | Accuracy, F1Score, Precision, Recall, AUROC, ConfusionMatrix, mAP, CorLoc |
| `datasets/` | ImageFolderDataset, COCOClassificationDataset, COCODetectionDataset |
| `optimizers/` | Adam, AdamW, SGD, RMSprop + schedulers |

## Adding a Component

1. Create the class in the appropriate submodule.
2. Decorate with `@registry.register("<Name>")`.
3. Import it in the submodule's `__init__.py`.

## Design Patterns

SimpleML is built on the **Modular Block** pattern: every component is a standalone block with a uniform interface, registered by name, and composed via config or fluent API — like snapping LEGO pieces together.

### Modular Block Rules

**1. Every block belongs to a category with its own Registry**
```python
# simpleml/registries.py
MODELS    = Registry("models")
LOSSES    = Registry("losses")
METRICS   = Registry("metrics")
# ...
```

**2. Every block registers itself with `@REGISTRY.register`**
```python
from simpleml.registries import MODELS

@MODELS.register
class MyModel(nn.Module):
    def __init__(self, num_classes: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

**3. Each category has a base interface (ABC) that every block respects**
- Models: `nn.Module` with `forward(x) -> Tensor`
- Metrics: `Metric` with `update(preds, targets)` and `compute()`
- Losses: `nn.Module` with `forward(preds, targets) -> Tensor`
- Datasets: `torch.utils.data.Dataset` with `__len__` and `__getitem__`

**4. Blocks are composed via config dict or fluent API — never coupled to each other**
```python
# Via YAML / dict
{"name": "MyModel", "params": {"num_classes": 10}}

# Via fluent API
API().model("MyModel", num_classes=10).loss("FocalLoss").fit()
```

**5. When creating a new block, always follow this order:**
1. Write the test in `tests/<category>/test_<name>.py` (TDD)
2. Create the class in the correct module, inheriting from the base interface
3. Decorate with `@REGISTRY.register`
4. Export it in the category's `__init__.py`

## Development Guidelines

- **All code, comments, docstrings, and tests must be written in English.**
- **TDD**: write tests before implementing. New features and bug fixes must have tests.
- Tests live in `tests/`. Mirror the `simpleml/` structure (e.g. `simpleml/metrics/accuracy.py` → `tests/metrics/test_accuracy.py`).
- All tests must pass before committing: `uv run pytest`.
- Keep tests fast and focused — use small synthetic tensors/fixtures, not real datasets.
- Lint and format before committing: `uv run ruff check . && uv run ruff format .`.

## Stack

Python 3.10+, torch, timm, albumentations, tensorboard, pyyaml, tqdm, numpy, scikit-learn. Package manager: `uv`.
