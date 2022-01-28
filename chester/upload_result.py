import sys
import os
import argparse
from chester import config
from chester.run_exp import rsync_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    remote_dir = config.REMOTE_DIR[args.mode]
    rsync_code(args.mode, remote_dir)
