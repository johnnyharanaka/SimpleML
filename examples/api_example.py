"""Example demonstrating the fluent API for building and running experiments."""

from simpleml import API

exp = (
    API()
    .model("TimmModel", model_name="resnet18", pretrained=True, num_classes=10)
    .loss("CrossEntropyLoss", label_smoothing=0.1)
    .optimizer("AdamW", lr=1e-3, weight_decay=1e-4)
    .scheduler("CosineAnnealingLR", T_max=20)
    .data(train="data/cifar10/train", val="data/cifar10/val", test="data/cifar10/test")
    .metrics("Accuracy", {"name": "F1Score", "params": {"average": "weighted"}})
    .train_config(epochs=20, batch_size=64, device="mps", save_best=True)
)

print("Config:", exp.to_config())

# Train
results = exp.fit()
print("Training results:", results)

# Evaluate on the test split using the best checkpoint
eval_results = exp.evaluate("test", checkpoint="checkpoints/best.pt")
print("Evaluation results:", eval_results)
