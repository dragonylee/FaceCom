import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from utils.completion import facial_mesh_completion
from utils.funcs import load_generator
from config.config import read_config


def main():
    parser = argparse.ArgumentParser(description='facial shape completion')

    parser.add_argument("--config_file", type=str)
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)

    parser.add_argument("--rr", type=bool, default=True)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--dis_percent", type=float, default=None)

    args = parser.parse_args()

    config = read_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = load_generator(config).to(device)
    facial_mesh_completion(args.in_file, args.out_file, config, generator, args.lambda_reg, args.verbose,
                           args.rr, args.dis_percent)


if __name__ == "__main__":
    main()
