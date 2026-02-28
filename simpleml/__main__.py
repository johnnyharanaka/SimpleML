"""Entry point for ``python -m simpleml``."""

from __future__ import annotations

import argparse
import sys

from simpleml import Trainer


def main(argv: list[str] | None = None) -> None:
    """Run training from a YAML config file.

    Usage::

        python -m simpleml config.yaml
        python -m simpleml config.yaml --resume checkpoints/last.pt
    """
    parser = argparse.ArgumentParser(
        prog="simpleml",
        description="SimpleML — train a model from a YAML config.",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    args = parser.parse_args(argv)

    trainer = Trainer.from_config(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    results = trainer.fit()

    print("\n--- Training complete ---")
    print(f"  Epochs trained : {results['epochs_trained']}")
    print(f"  Train loss     : {results['last_train_loss']:.4f}")
    if results.get("last_val_loss") is not None:
        print(f"  Val loss       : {results['last_val_loss']:.4f}")
    if results.get("best_val_loss") is not None and results["best_val_loss"] < float("inf"):
        print(f"  Best val loss  : {results['best_val_loss']:.4f}")
    if results.get("last_metrics"):
        print("  Metrics:")
        for name, value in results["last_metrics"].items():
            print(f"    {name}: {value:.4f}")


if __name__ == "__main__":
    main()