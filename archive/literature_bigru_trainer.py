import argparse

import ucddb_literature_train_common as common


def main():
    parser = argparse.ArgumentParser(description="Chen-style UCDDB CNN-BiGRU baseline using RRI and R-peak amplitude.")
    common.add_train_args(parser)
    args = parser.parse_args()
    common.run_training(args, model_kind="bigru", include_raw=False)


if __name__ == "__main__":
    main()
