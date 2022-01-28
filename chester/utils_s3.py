from chester import config_ec2 as config
from io import StringIO
import base64
import os
import os.path as osp
import subprocess
import hashlib
import datetime
import dateutil
import re

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')


def dedent(s):
    lines = [l.strip() for l in s.split('\n')]
    return '\n'.join(lines)


def upload_file_to_s3(script_content):
    import tempfile
    import uuid
    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(script_content.encode())
    f.close()
    remote_path = os.path.join(
        config.AWS_CODE_SYNC_S3_PATH, "oversize_bash_scripts", str(uuid.uuid4()))
    subprocess.check_call(["aws", "s3", "cp", f.name, remote_path])
    os.unlink(f.name)
    return remote_path


S3_CODE_PATH = None


def s3_sync_code(config, dry=False):
    global S3_CODE_PATH
    if S3_CODE_PATH is not None:
        return S3_CODE_PATH
    base = config.AWS_CODE_SYNC_S3_PATH
    has_git = True

    if config.FAST_CODE_SYNC:
        try:
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        except subprocess.CalledProcessError as _:
            print("Warning: failed to execute git commands")
            current_commit = None
        file_name = str(timestamp) + "_" + hashlib.sha224(
            subprocess.check_output(["pwd"]) + str(current_commit).encode() + str(timestamp).encode()
        ).hexdigest() + ".tar.gz"

        file_path = "/tmp/" + file_name

        tar_cmd = ["tar", "-zcvf", file_path, "-C", config.PROJECT_PATH]
        for pattern in config.FAST_CODE_SYNC_IGNORES:
            tar_cmd += ["--exclude", pattern]
        tar_cmd += ["-h", "."]

        remote_path = "%s/%s" % (base, file_name)

        upload_cmd = ["aws", "s3", "cp", file_path, remote_path]

        mujoco_key_cmd = [
            "aws", "s3", "sync", config.MUJOCO_KEY_PATH, "{}/.mujoco/".format(base)]

        print(" ".join(tar_cmd))
        print(" ".join(upload_cmd))
        print(" ".join(mujoco_key_cmd))

        if not dry:
            subprocess.check_call(tar_cmd)
            subprocess.check_call(upload_cmd)
            try:
                subprocess.check_call(mujoco_key_cmd)
            except Exception as e:
                print(e)

        S3_CODE_PATH = remote_path
        return remote_path
    else:
        try:
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            clean_state = len(
                subprocess.check_output(["git", "status", "--porcelain"])) == 0
        except subprocess.CalledProcessError as _:
            print("Warning: failed to execute git commands")
            has_git = False
        dir_hash = base64.b64encode(subprocess.check_output(["pwd"])).decode("utf-8")
        code_path = "%s_%s" % (
            dir_hash,
            (current_commit if clean_state else "%s_dirty_%s" % (current_commit, timestamp)) if
            has_git else timestamp
        )
        full_path = "%s/%s" % (base, code_path)
        cache_path = "%s/%s" % (base, dir_hash)
        cache_cmds = ["aws", "s3", "cp", "--recursive"] + \
                     flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
                     [cache_path, full_path]
        cmds = ["aws", "s3", "cp", "--recursive"] + \
               flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
               [".", full_path]
        caching_cmds = ["aws", "s3", "cp", "--recursive"] + \
                       flatten(["--exclude", "%s" % pattern] for pattern in config.CODE_SYNC_IGNORES) + \
                       [full_path, cache_path]
        mujoco_key_cmd = [
            "aws", "s3", "sync", config.MUJOCO_KEY_PATH, "{}/.mujoco/".format(base)]
        print(cache_cmds, cmds, caching_cmds, mujoco_key_cmd)
        if not dry:
            subprocess.check_call(cache_cmds)
            subprocess.check_call(cmds)
            subprocess.check_call(caching_cmds)
            try:
                subprocess.check_call(mujoco_key_cmd)
            except Exception:
                print('Unable to sync mujoco keys!')
        S3_CODE_PATH = full_path
        return full_path


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))


def to_local_command(params, python_command="python", script=osp.join(config.PROJECT_PATH, 'scripts/run_experiment.py'), use_gpu=False):
    command = python_command + " " + script
    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)
    if post_commands is not None:
        print("Not executing the post_commands: ", post_commands)

    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += \
                        "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    for pre_command in reversed(pre_commands):
        command = pre_command + " && " + command
    return command


def launch_ec2(params_list, exp_prefix, docker_image, code_full_path,
               python_command="python",
               script='scripts/run_experiment.py',
               aws_config=None, dry=False, terminate_machine=True, use_gpu=False, sync_s3_pkl=False,
               sync_s3_png=False,
               sync_s3_log=False,
               sync_s3_html=False,
               sync_s3_mp4=False,
               sync_s3_gif=False,
               sync_s3_pth=False,
               sync_s3_txt=False,
               sync_log_on_termination=True,
               periodic_sync=True, periodic_sync_interval=15):
    if len(params_list) == 0:
        return

    default_config = dict(
        image_id=config.AWS_IMAGE_ID,
        instance_type=config.AWS_INSTANCE_TYPE,
        key_name=config.AWS_KEY_NAME,
        spot=config.AWS_SPOT,
        spot_price=config.AWS_SPOT_PRICE,
        iam_instance_profile_name=config.AWS_IAM_INSTANCE_PROFILE_NAME,
        security_groups=config.AWS_SECURITY_GROUPS,
        security_group_ids=config.AWS_SECURITY_GROUP_IDS,
        network_interfaces=config.AWS_NETWORK_INTERFACES,
        instance_interruption_behavior='terminate',  # TODO
    )

    if aws_config is None:
        aws_config = dict()
    aws_config = dict(default_config, **aws_config)

    sio = StringIO()
    sio.write("#!/bin/bash\n")
    sio.write("{\n")
    sio.write("""
        die() { status=$1; shift; echo "FATAL: $*"; exit $status; }
    """)
    sio.write("""
        EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id`"
    """)
    # sio.write("""service docker start""")
    # sio.write("""docker --config /home/ubuntu/.docker pull {docker_image}""".format(docker_image=docker_image))
    sio.write("""
        export PATH=/home/ubuntu/bin:/home/ubuntu/.local/bin:$PATH
    """)
    sio.write("""
        export PATH=/home/ubuntu/miniconda3/bin:/usr/local/cuda/bin:$PATH
    """)
    sio.write("""
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
    """)
    sio.write("""
        echo $PATH
    """)
    sio.write("""
        export AWS_DEFAULT_REGION={aws_region}
    """.format(aws_region=config.AWS_BUCKET_REGION_NAME))  # add AWS_BUCKET_REGION_NAME=us-east-1 in your config.py
    sio.write("""
        pip install --upgrade --user awscli
    """)

    sio.write("""
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
    """.format(exp_name=params_list[0].get("exp_name"), aws_region=config.AWS_REGION_NAME))
    if config.LABEL:
        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=owner,Value={label} --region {aws_region}
        """.format(label=config.LABEL, aws_region=config.AWS_REGION_NAME))
    sio.write("""
        aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=exp_prefix,Value={exp_prefix} --region {aws_region}
    """.format(exp_prefix=exp_prefix, aws_region=config.AWS_REGION_NAME))

    if config.FAST_CODE_SYNC:
        sio.write("""
            aws s3 cp {code_full_path} /tmp/chester_code.tar.gz
        """.format(code_full_path=code_full_path, local_code_path=config.CODE_DIR))
        sio.write("""
            mkdir -p {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.CODE_DIR,
                   aws_region=config.AWS_REGION_NAME))
        sio.write("""
            tar -zxvf /tmp/chester_code.tar.gz -C {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.CODE_DIR,
                   aws_region=config.AWS_REGION_NAME))
    else:
        sio.write("""
            aws s3 cp --recursive {code_full_path} {local_code_path}
        """.format(code_full_path=code_full_path, local_code_path=config.CODE_DIR))
    s3_mujoco_key_path = config.AWS_CODE_SYNC_S3_PATH + '/.mujoco/'
    sio.write("""
        aws s3 cp --recursive {} {}
    """.format(s3_mujoco_key_path, config.MUJOCO_KEY_PATH))
    sio.write("""
        cd {local_code_path}
    """.format(local_code_path=config.CODE_DIR))

    for params in params_list:
        log_dir = params.get("log_dir")
        remote_log_dir = params.pop("remote_log_dir")
        env = params.pop("env", None)

        sio.write("""
            aws ec2 create-tags --resources $EC2_INSTANCE_ID --tags Key=Name,Value={exp_name} --region {aws_region}
        """.format(exp_name=params.get("exp_name"), aws_region=config.AWS_REGION_NAME))
        sio.write("""
            mkdir -p {log_dir}
        """.format(log_dir=log_dir))
        if periodic_sync:
            include_png = " --include '*.png' " if sync_s3_png else " "
            include_pkl = " --include '*.pkl' " if sync_s3_pkl else " "
            include_log = " --include '*.log' " if sync_s3_log else " "
            include_html = " --include '*.html' " if sync_s3_html else " "
            include_mp4 = " --include '*.mp4' " if sync_s3_mp4 else " "
            include_gif = " --include '*.gif' " if sync_s3_gif else " "
            include_pth = " --include '*.pth' " if sync_s3_pth else " "
            include_txt = " --include '*.txt' " if sync_s3_txt else " "
            sio.write("""
                while /bin/true; do
                    aws s3 sync --exclude '*' {include_png} {include_pkl} {include_log} {include_html} {include_mp4} {include_gif} {include_pth} {include_txt} --include '*.csv' --include '*.json' {log_dir} {remote_log_dir}
                    sleep {periodic_sync_interval}
                done & echo sync initiated""".format(include_png=include_png, include_pkl=include_pkl,
                                                     include_log=include_log, include_html=include_html,
                                                     include_mp4=include_mp4, include_gif=include_gif,
                                                     include_pth=include_pth, include_txt=include_txt,
                                                     log_dir=log_dir, remote_log_dir=remote_log_dir,
                                                     periodic_sync_interval=periodic_sync_interval))
            if sync_log_on_termination:
                sio.write("""
                    while /bin/true; do
                        if [ -z $(curl -Is http://169.254.169.254/latest/meta-data/spot/termination-time | head -1 | grep 404 | cut -d \  -f 2) ]
                          then
                            logger "Running shutdown hook."
                            aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log
                            aws s3 cp --recursive {log_dir} {remote_log_dir}
                            break
                          else
                            # Spot instance not yet marked for termination.
                            sleep 5
                        fi
                    done & echo log sync initiated
                """.format(log_dir=log_dir, remote_log_dir=remote_log_dir))
        sio.write("""{command}""".format(command=to_local_command(params, python_command=python_command, script=script, use_gpu=use_gpu)))
        sio.write("""
        aws s3 cp --recursive {log_dir} {remote_log_dir}
         """.format(log_dir=log_dir, remote_log_dir=remote_log_dir))
        sio.write("""
        aws s3 cp /home/ubuntu/user_data.log {remote_log_dir}/stdout.log
        """.format(remote_log_dir=remote_log_dir))

    if terminate_machine:
        sio.write("""
            EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
            aws ec2 terminate-instances --instance-ids $EC2_INSTANCE_ID --region {aws_region}
        """.format(aws_region=config.AWS_REGION_NAME))
    sio.write("} >> /home/ubuntu/user_data.log 2>&1\n")

    full_script = dedent(sio.getvalue())

    import boto3
    import botocore
    if aws_config["spot"]:
        ec2 = boto3.client(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
    else:
        ec2 = boto3.resource(
            "ec2",
            region_name=config.AWS_REGION_NAME,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )

    print("len_full_script", len(full_script))
    if len(full_script) > 16384 or len(base64.b64encode(full_script.encode()).decode("utf-8")) > 16384:
        # Script too long; need to upload script to s3 first.
        # We're being conservative here since the actual limit is 16384 bytes
        s3_path = upload_file_to_s3(full_script)
        sio = StringIO()
        sio.write("#!/bin/bash\n")
        sio.write("""
        aws s3 cp {s3_path} /home/ubuntu/remote_script.sh --region {aws_region} && \\
        chmod +x /home/ubuntu/remote_script.sh && \\
        bash /home/ubuntu/remote_script.sh
        """.format(s3_path=s3_path, aws_region=config.AWS_REGION_NAME))
        user_data = dedent(sio.getvalue())
    else:
        user_data = full_script
    print(full_script)
    with open("/tmp/full_script", "w") as f:
        f.write(full_script)

    instance_args = dict(
        ImageId=aws_config["image_id"],
        KeyName=aws_config["key_name"],
        UserData=user_data,
        InstanceType=aws_config["instance_type"],
        EbsOptimized=config.EBS_OPTIMIZED,
        SecurityGroups=aws_config["security_groups"],
        SecurityGroupIds=aws_config["security_group_ids"],
        NetworkInterfaces=aws_config["network_interfaces"],
        IamInstanceProfile=dict(
            Name=aws_config["iam_instance_profile_name"],
        ),
        **config.AWS_EXTRA_CONFIGS,
    )

    if len(instance_args["NetworkInterfaces"]) > 0:
        # disable_security_group = query_yes_no(
        #     "Cannot provide both network interfaces and security groups info. Do you want to disable security group settings?",
        #     default="yes",
        # )
        disable_security_group = True
        if disable_security_group:
            instance_args.pop("SecurityGroups")
            instance_args.pop("SecurityGroupIds")

    if aws_config.get("placement", None) is not None:
        instance_args["Placement"] = aws_config["placement"]
    if not aws_config["spot"]:
        instance_args["MinCount"] = 1
        instance_args["MaxCount"] = 1
    print("************************************************************")
    print(instance_args["UserData"])
    print("************************************************************")
    if aws_config["spot"]:
        instance_args["UserData"] = base64.b64encode(instance_args["UserData"].encode()).decode("utf-8")
        spot_args = dict(
            DryRun=dry,
            InstanceCount=1,
            LaunchSpecification=instance_args,
            SpotPrice=aws_config["spot_price"],
            # ClientToken=params_list[0]["exp_name"],
        )
        import pprint
        pprint.pprint(spot_args)
        if not dry:
            response = ec2.request_spot_instances(**spot_args)
            print(response)
            spot_request_id = response['SpotInstanceRequests'][
                0]['SpotInstanceRequestId']
            for _ in range(10):
                try:
                    ec2.create_tags(
                        Resources=[spot_request_id],
                        Tags=[
                            {'Key': 'Name', 'Value': params_list[0]["exp_name"]}
                        ],
                    )
                    break
                except botocore.exceptions.ClientError:
                    continue
    else:
        import pprint
        pprint.pprint(instance_args)
        ec2.q(
            DryRun=dry,
            **instance_args
        )
