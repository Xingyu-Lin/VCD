import os
import subprocess
import argparse

def aws_sync(bucket_name, s3_log_dir, target_dir, args):
    cmd = 'aws s3 cp --recursive s3://%s/%s %s' % (bucket_name, s3_log_dir, target_dir)
    exlus = ['"*.pkl"', '"*.gif"', '"*.png"', '"*.pth"']
    inclus = []
    if args.gif:
        exlus.remove('"*.gif"')
    if args.png:
        exlus.remove('"*.png"')
    if args.param:
        inclus.append('"params.pkl"')
        exlus.remove('"*.pkl"')

    if not args.include_all:
        for exc in exlus:
            cmd += ' --exclude ' + exc
        
        for inc in inclus:
            cmd += ' --include ' + inc
    
    print(cmd)
    # exit()
    subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str, help='S3 Log dir')
    parser.add_argument('-b', '--bucket', type=str, default='chester-softgym', help='S3 Bucket')
    parser.add_argument('--param', type=int, default=0, help='Exclude')
    parser.add_argument('--gif', type=int, default=0, help='Exclude')
    parser.add_argument('--png', type=int, default=0, help='Exclude')
    parser.add_argument('--include_all', type=int, default=1, help='pull all data')

    args = parser.parse_args()
    s3_log_dir = "rllab/experiments/" + args.log_dir
    local_dir = os.path.join('./data', 'corl_s3_data', args.log_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    aws_sync(args.bucket, s3_log_dir, local_dir, args)


if __name__ == "__main__":
    main()
