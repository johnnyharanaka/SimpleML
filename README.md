# SimpleML

A modular, registry-based ML training framework. Define your entire training pipeline in a YAML config and run it in one command.

```python
from simpleml import Trainer

trainer = Trainer.from_config("config.yaml")
trainer.fit()
```

## Installation

Requires Python 3.10+.

```bash
# Clone the repo
git clone https://github.com/your-username/SimpleML.git
cd SimpleML

# Install with uv
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Organize your data

SimpleML expects image classification data in folder structure:

```
data/
├── train/
│   ├── class_a/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class_b/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class_a/
    └── class_b/
```

### 2. Write a config

```yaml
model:
  name: TimmModel
  params:
    model_name: resnet18
    pretrained: true
    num_classes: 10

dataset:
  train:
    name: ImageFolderDataset
    params:
      root: data/train
  val:
    name: ImageFolderDataset
    params:
      root: data/val

loss:
  name: CrossEntropyLoss

optimizer:
  name: AdamW
  params:
    lr: 1e-3
    weight_decay: 1e-4

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 20

metrics:
  - name: Accuracy
  - name: F1Score
    params:
      average: weighted

training:
  epochs: 20
  batch_size: 64
  device: mps
```

See [`examples/`](examples/) for more config examples.

### 3. Train

**From the command line:**

```bash
python -m simpleml config.yaml
```

To resume from a checkpoint:

```bash
python -m simpleml config.yaml --resume checkpoints/last.pt
```

**From Python:**

```python
from simpleml import Trainer

trainer = Trainer.from_config("config.yaml")
results = trainer.fit()
```

## Available Components

All components are registered by name and selected via config.

| Category    | Components                                                       |
|-------------|------------------------------------------------------------------|
| Models      | `TimmModel` (any [timm](https://github.com/huggingface/pytorch-image-models) architecture) |
| Losses      | `CrossEntropyLoss`, `BCEWithLogitsLoss`, `FocalLoss`, `NTXentLoss`, `SupConLoss`, `TripletMarginLoss` |
| Optimizers  | `Adam`, `AdamW`, `SGD`, `RMSprop`                                |
| Schedulers  | `StepLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `OneCycleLR` |
| Metrics     | `Accuracy`, `F1Score`, `Precision`, `Recall`, `AUROC`, `ConfusionMatrix` |
| Datasets    | `ImageFolderDataset`                                             |

## Training Features

- Automatic device selection (MPS > CUDA > CPU)
- Mixed precision training (CUDA)
- Gradient clipping (norm and value)
- TensorBoard logging
- Best and last checkpoint saving
- Resume from checkpoint
- Configurable validation frequency

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## License

Apache 2.0
