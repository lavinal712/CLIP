import argparse
import ast


def parse_args(args):
    parser = argparse.ArgumentParser()

    args = parser.parse_args(args)

    return args
