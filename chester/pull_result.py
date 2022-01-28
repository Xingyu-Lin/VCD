import sys
import os
import argparse

sys.path.append('.')
from chester import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('folder', type=str)
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    parser.add_argument('--img', action='store_true', default=False)
    parser.add_argument('--pkl', action='store_true', default=False)
    parser.add_argument('--gif', action='store_true', default=False)
    parser.add_argument('--newdatadir', action='store_true', default=False)
    args = parser.parse_args()

    args.folder = args.folder.rstrip('/')
    if args.folder.rfind('/') !=-1:
        local_dir = os.path.join('./data', args.host, args.folder[:args.folder.rfind('/')])
    else:
        local_dir = os.path.join('./data', args.host)
    # if args.newdatadir:
    dir_path = '/data/yufeiw2/softagent_prvil_merge/'
    # else:
        # dir_path = config.REMOTE_DIR[args.host]
    remote_data_dir = os.path.join(dir_path, 'data', 'local', args.folder)
    command = """rsync -avzh --delete --progress {host}:{remote_data_dir} {local_dir}""".format(host=args.host,
                                                                                                remote_data_dir=remote_data_dir,
                                                                                                local_dir=local_dir)
    if args.bare:
        command += """  --exclude '*checkpoin*' --exclude '*ckpt*'  --exclude '*tfevents*' --exclude '*.pth' --exclude '*.pt' --include '*.csv' --include '*.json' --delete"""
    if not args.img:
        command += """ --exclude '*.png' """
    if not args.gif:
        command += """ --exclude '*.gif' """
    if not args.pkl:
        command += """ --exclude '*.pkl'  """
    if args.dry:
        print(command)
    else:
        os.system(command)
