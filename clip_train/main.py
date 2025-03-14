import os
import sys

from clip_train.data import COCODataset
from clip_train.params import parse_args


def main(args):
    args = parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
