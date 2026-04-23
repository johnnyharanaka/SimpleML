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

SimpleML é construído sobre o pattern **Modular Block**: cada componente é um bloco independente com interface uniforme, registrado por nome, e montado via config ou fluent API — igual a encaixar peças de LEGO.

### Regras do Modular Block

**1. Todo bloco pertence a uma categoria com sua própria Registry**
```python
# simpleml/registries.py
MODELS    = Registry("models")
LOSSES    = Registry("losses")
METRICS   = Registry("metrics")
# ...
```

**2. Todo bloco se registra com `@REGISTRY.register`**
```python
from simpleml.registries import MODELS

@MODELS.register
class MeuModelo(nn.Module):
    def __init__(self, num_classes: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

**3. Cada categoria tem uma interface base (ABC) que todos os blocos respeitam**
- Modelos: `nn.Module` com `forward(x) -> Tensor`
- Métricas: `Metric` com `update(preds, targets)` e `compute()`
- Losses: `nn.Module` com `forward(preds, targets) -> Tensor`
- Datasets: `torch.utils.data.Dataset` com `__len__` e `__getitem__`

**4. Blocos são compostos por config dict ou fluent API — nunca acoplados entre si**
```python
# Via YAML / dict
{"name": "MeuModelo", "params": {"num_classes": 10}}

# Via fluent API
API().model("MeuModelo", num_classes=10).loss("FocalLoss").fit()
```

**5. Ao criar um novo bloco, siga sempre esta ordem:**
1. Escreva o teste em `tests/<categoria>/test_<nome>.py` (TDD)
2. Crie a classe no módulo correto herdando da interface base
3. Decore com `@REGISTRY.register`
4. Exporte no `__init__.py` da categoria

## Development Guidelines

- **TDD**: write tests before implementing. New features and bug fixes must have tests.
- Tests live in `tests/`. Mirror the `simpleml/` structure (e.g. `simpleml/metrics/accuracy.py` → `tests/metrics/test_accuracy.py`).
- All tests must pass before committing: `uv run pytest`.
- Keep tests fast and focused — use small synthetic tensors/fixtures, not real datasets.
- Lint and format before committing: `uv run ruff check . && uv run ruff format .`.

## Stack

Python 3.10+, torch, timm, albumentations, tensorboard, pyyaml, tqdm, numpy, scikit-learn. Package manager: `uv`.
