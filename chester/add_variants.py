# Add variants to finished experiments
import argparse
import os
import json
from pydoc import locate
import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_folder', type=str, help='Root of the experiment folder to walk through')
    parser.add_argument('key', type=str, help='Name of the additional key')
    parser.add_argument('value', help='Value of the additional key')
    parser.add_argument('value_type', default='str', type=str, help='Type of the additional key')
    parser.add_argument('remote', nargs='?', default=None, type=str, ) # Optional

    args = parser.parse_args()
    exp_paths = [x[0] for x in os.walk(args.exp_folder, followlinks=True)]

    value_type = locate(args.value_type)
    if value_type == bool:
        value = args.value in ['1', 'True', 'true']
    else:
        value = value_type(args.value)

    for exp_path in exp_paths:
        try:
            variant_path = os.path.join(exp_path, "variant.json")
            # Modify locally
            with open(variant_path, 'r') as f:
                vv = json.load(f)
            if args.key in vv:
                print('Warning: key already in variants. {} = {}. Setting it to {}'.format(args.key, vv[args.key], value))

            vv[args.key] = value
            with open(variant_path, 'w') as f:
                json.dump(vv, f, indent=2, sort_keys=True)
            print('{} modified'.format(variant_path))

            # Upload it to remote
            if args.remote is not None:
                p = variant_path.rstrip('/').split('/')
                sub_exp_name, exp_name = p[-2], p[-3]

                remote_dir = os.path.join(config.REMOTE_DIR[args.remote], 'data', 'local', exp_name, sub_exp_name, 'variant.json')
                os.system('scp {} {}:{}'.format(variant_path, args.remote, remote_dir))
        except IOError as e:
            print(e)


if __name__ == '__main__':
    main()
