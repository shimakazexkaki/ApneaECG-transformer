import argparse

import ucddb_literature_train_common as common


def main():
    parser = argparse.ArgumentParser(description="Hybrid UCDDB raw-ECG + RRI Transformer trainer.")
    common.add_train_args(parser)
    args = parser.parse_args()
    common.run_training(args, model_kind="hybrid_transformer", include_raw=True)


if __name__ == "__main__":
    main()
