import argparse

import ucddb_literature_train_common as common


def main():
    parser = argparse.ArgumentParser(description="UCDDB Transformer experiments on RRI/R-peak-amplitude features.")
    common.add_train_args(parser)
    parser.add_argument(
        "--variant",
        choices=["rri_transformer", "cnn_transformer"],
        default="rri_transformer",
        help="Use a pure RRI Transformer or a CNN front-end followed by Transformer.",
    )
    args = parser.parse_args()
    common.run_training(args, model_kind=args.variant, include_raw=False)


if __name__ == "__main__":
    main()
