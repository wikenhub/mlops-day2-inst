"""Model training entrypoint (SageMaker Training)."""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val")
    p.add_argument("--model-dir", default="/opt/ml/model")
    p.add_argument("--output-data-dir", default="/opt/ml/output/data")
    # args = p.parse_args()
    print("Train placeholder – will implement in Lab 4.")


if __name__ == "__main__":
    main()
