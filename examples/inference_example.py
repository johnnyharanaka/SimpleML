"""Inference examples — how to run predictions and evaluate with SimpleML."""

from simpleml import Predictor

# ---------------------------------------------------------------
# 1. Avaliar no test set direto do config
# ---------------------------------------------------------------

predictor = Predictor.from_config(
    "examples/inference.yaml",
    checkpoint="checkpoints/best.pt",
)

results = predictor.evaluate_from_config("test")

print("--- Test set ---")
print(f"  Samples: {results['predictions'].logits.shape[0]}")
for name, value in results["metrics"].items():
    print(f"  {name}: {value:.4f}")

# ---------------------------------------------------------------
# 2. Avaliar no val set (mesmo config, split diferente)
# ---------------------------------------------------------------

results_val = predictor.evaluate_from_config("val")

print("\n--- Val set ---")
for name, value in results_val["metrics"].items():
    print(f"  {name}: {value:.4f}")

# ---------------------------------------------------------------
# 3. Predição em uma única imagem
# ---------------------------------------------------------------

result = predictor.predict_image("data/cifar10/test/cat/img001.jpg")

print("\n--- Single image ---")
print(f"  Predicted class: {result.predicted_classes.item()}")
print(f"  Probabilities:   {result.probabilities.squeeze().tolist()}")

# ---------------------------------------------------------------
# 4. Predição em batch (diretório de imagens)
# ---------------------------------------------------------------

result = predictor.predict_batch("data/cifar10/test/cat/")

print("\n--- Batch (directory) ---")
print(f"  Predicted classes: {result.predicted_classes.tolist()}")

# ---------------------------------------------------------------
# 5. Usar o config de treino diretamente (reutiliza o mesmo YAML)
# ---------------------------------------------------------------

predictor2 = Predictor.from_config(
    "examples/cifar10.yaml",
    checkpoint="checkpoints/best.pt",
)

results2 = predictor2.evaluate_from_config("val")

print("\n--- Val set (from training config) ---")
for name, value in results2["metrics"].items():
    print(f"  {name}: {value:.4f}")
