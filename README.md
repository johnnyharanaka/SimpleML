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

## Fluent API (no YAML required)

Instead of a config file, you can wire everything together in Python using the `API` builder:

```python
from simpleml import API

exp = (
    API()
    .model("TimmModel", model_name="resnet18", pretrained=True, num_classes=10)
    .loss("CrossEntropyLoss", label_smoothing=0.1)
    .optimizer("AdamW", lr=1e-3, weight_decay=1e-4)
    .scheduler("CosineAnnealingLR", T_max=20)
    .data(train="data/train", val="data/val")
    .metrics("Accuracy", {"name": "F1Score", "params": {"average": "weighted"}})
    .train_config(epochs=20, batch_size=64, device="mps", save_best=True, best_filename="best.pth")
    .fit()
)
```

### `.model(name, **params)`

Selects the model by registry name and passes keyword arguments as params.

```python
.model("TimmModel", model_name="resnet18", pretrained=True, num_classes=10)
```

### `.loss(name, **params)`

```python
.loss("CrossEntropyLoss", label_smoothing=0.1)
```

### `.optimizer(name, **params)` and `.scheduler(name, **params)`

```python
.optimizer("AdamW", lr=1e-3, weight_decay=1e-4)
.scheduler("CosineAnnealingLR", T_max=20)
```

### `.data(train, val, test, dataset)`

Each split accepts either:
- A **string** — shorthand for `ImageFolderDataset` with that root path.
- A **dict** — full component spec `{"name": ..., "params": {...}}`, for any dataset with transforms.

```python
# Simple — string shorthand
.data(train="data/train", val="data/val")

# Full spec with transforms
.data(
    train={
        "name": "ImageFolderDataset",
        "params": {
            "root": "data/train",
            "transform": [
                {"name": "Resize", "params": {"height": 224, "width": 224}},
                {"name": "HorizontalFlip"},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"name": "ToTensorV2"},
            ],
        },
    },
    val={
        "name": "COCOClassificationDataset",
        "params": {
            "root": "data/val",
            "classes": ["Cat", "Dog"],
            "default_class": "Dog",
            "transform": [
                {"name": "Resize", "params": {"height": 224, "width": 224}},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
                {"name": "ToTensorV2"},
            ],
        },
    },
)
```

### `.metrics(*names_or_configs)`

Accepts metric names (strings) or full dicts with params.

```python
.metrics("Accuracy", "F1Score")
.metrics("Accuracy", {"name": "F1Score", "params": {"average": "weighted"}})
```

### `.train_config(**kwargs)` and `.infer_config(**kwargs)`

Merge any key into the `training` / `inference` config sections.

```python
.train_config(
    epochs=20,
    batch_size=64,
    device="mps",
    save_best=True,
    save_last=True,
    best_filename="best.pth",
    checkpoint_dir="checkpoints",
    log_dir="runs",
    val_every=1,
    best_metric="Accuracy",   # optional: metric to monitor for best checkpoint
    best_metric_mode="max",   # "max" (accuracy, f1) or "min" (loss-like metrics)
)
.infer_config(batch_size=32, device="mps")
```

### `.fit()`, `.evaluate()`, `.predict_image()`, `.predict_batch()`

```python
# Train
results = exp.fit()

# Evaluate on test split
eval_results = exp.evaluate("test", checkpoint="checkpoints/best.pth")
print(eval_results["metrics"]["Accuracy"])

# Inference
result = exp.predict_image("path/to/image.jpg", checkpoint="checkpoints/best.pth")
results = exp.predict_batch("path/to/images/", checkpoint="checkpoints/best.pth")
```

### Loading from YAML

You can also load an existing YAML file into an `API` instance and override settings programmatically:

```python
exp = API.from_yaml("configs/resnet.yaml")
exp.fit(epochs=5)  # override epochs for this run only
```

---

## Available Components

All components are registered by name and selected via config.

| Category    | Components                                                       |
|-------------|------------------------------------------------------------------|
| Models      | `TimmModel` (any [timm](https://github.com/huggingface/pytorch-image-models) architecture), `DinoClassifier`, `LoRADinoClassifier` |
| Losses      | `CrossEntropyLoss`, `BCEWithLogitsLoss`, `FocalLoss`, `NTXentLoss`, `SupConLoss`, `TripletMarginLoss` |
| Optimizers  | `Adam`, `AdamW`, `SGD`, `RMSprop`                                |
| Schedulers  | `StepLR`, `CosineAnnealingLR`, `CosineAnnealingWarmRestarts`, `OneCycleLR` |
| Metrics     | `Accuracy`, `F1Score`, `Precision`, `Recall`, `AUROC`, `ConfusionMatrix` |
| Datasets    | `ImageFolderDataset`, `COCOClassificationDataset`                |

## Best Checkpoint by Metric

By default, the best checkpoint is saved based on the lowest `val_loss`. To monitor a specific metric instead, set `best_metric` in the training config:

```python
# Via fluent API
.train_config(best_metric="Accuracy", best_metric_mode="max")

# Via YAML
# training:
#   best_metric: Accuracy
#   best_metric_mode: max  # "max" for higher-is-better, "min" for lower-is-better
```

The metric name must match the class name of the metric registered in `.metrics()` (e.g. `Accuracy`, `F1Score`, `AUROC`). When `best_metric` is not set, the trainer falls back to `val_loss`.

## Training Features

- Automatic device selection (MPS > CUDA > CPU)
- Mixed precision training (CUDA)
- Gradient clipping (norm and value)
- TensorBoard logging
- Best and last checkpoint saving (by `val_loss` or any metric)
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
