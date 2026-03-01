from simpleml import API

def main():
    exp = API.from_yaml("configs/dino_detection.yaml")

    exp.fit()

    eval_results = exp.evaluate("test", checkpoint="checkpoints/dino.pth")
    print("Evaluation results:")
    print(f"mAP:    {eval_results['metrics']['MeanAveragePrecision']:.4f}")
    print(f"CorLoc: {eval_results['metrics']['CorLoc']:.4f}")

if __name__ == "__main__":
    main()
