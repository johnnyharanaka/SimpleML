from simpleml import API

def main():
    exp = API.from_yaml("configs/dino.yaml")

    exp.fit()

    eval_results = exp.evaluate("test", checkpoint="checkpoints/dino.pth")
    print("Evaluation results:")
    print(f"Accuracy: {eval_results['metrics']['Accuracy']}")

if __name__ == "__main__":
  main()