"""Feature preprocessing job entrypoint (SageMaker Processing)."""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    # args = p.parse_args()
    print("Preprocess placeholder we will implement in Lab 3.")


if __name__ == "__main__":
    main()
