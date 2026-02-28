from simpleml import API

def main():
    exp = API.from_yaml("configs/resnet.yaml")

    results = exp.fit()
    print("Training results:", results)

    eval_results = exp.evaluate("test", checkpoint="checkpoints/resnet.pth")
    print("Evaluation results:", eval_results)

if __name__ == "__main__":
  main()