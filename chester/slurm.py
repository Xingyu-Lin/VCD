import os
import os.path as osp
import re
from subprocess import run
from tempfile import NamedTemporaryFile
from chester import config

# TODO remove the singularity part

slurm_dir = './'


def slurm_run_scripts(scripts):
    """this is another function that those _sub files should call. this actually execute files"""
    # TODO support running multiple scripts

    assert isinstance(scripts, str)

    os.chdir(slurm_dir)

    # make sure it will run.
    assert scripts.startswith('#!/usr/bin/env bash\n')
    file_temp = NamedTemporaryFile(delete=False)
    file_temp.write(scripts.encode('utf-8'))
    file_temp.close()
    run(['sbatch', file_temp.name], check=True)
    os.remove(file_temp.name)


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


def to_slurm_command(params, header, python_command="python", remote_dir='~/',
                     script=osp.join(config.PROJECT_PATH, 'scripts/run_experiment.py'),
                     simg_dir=None, use_gpu=False, modules=None, cuda_module=None, use_singularity=True,
                     mount_options=None, compile_script=None, wait_compile=None, set_egl_gpu=False):
    # TODO Add code for specifying the resource allocation
    # TODO Check if use_gpu can be applied
    """
    Transfer the commands to the format that can be run by slurm.
    :param params:
    :param python_command:
    :param script:
    :param use_gpu:
    :return:
    """
    assert simg_dir is not None
    command = python_command + " " + script

    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)

    command_list = list()
    command_list.append(header)

    # Log into singularity shell
    if use_singularity:
        command_list.append('set -x')  # echo commands to stdout
        command_list.append('set -u')  # throw an error if unset variable referenced
        command_list.append('set -e')  # exit on errors
        command_list.append('srun hostname')

        for remote_module in modules:
            command_list.append('module load ' + remote_module)
        if use_gpu:
            assert cuda_module is not None
            command_list.append('module load ' + cuda_module)
        command_list.append('cd {}'.format(remote_dir))
        # First execute a bash program inside the container and then run all the following commands

        if mount_options is not None:
            options = '-B ' + mount_options
        else:
            options = ''
        sing_prefix = 'singularity exec {} {} {} /bin/bash -c'.format(options, '--nv' if use_gpu else '', simg_dir)
        sing_commands = list()
        if compile_script is None or 'prepare' not in compile_script :
            # sing_commands.append('. ./prepare_1.0.sh')
            sing_commands.append('. ./prepare.sh')
        if set_egl_gpu:
            sing_commands.append('export EGL_GPU=$SLURM_JOB_GRES')
            sing_commands.append('echo $EGL_GPU')
        if compile_script is not None:
            sing_commands.append('./' + compile_script)
        if wait_compile is not None:
            sing_commands.append('sleep '+str(int(wait_compile)))

    if pre_commands is not None:
        command_list.extend(pre_commands)
    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    sing_commands.append(command)
    all_sing_cmds = ' && '.join(sing_commands)
    command_list.append(sing_prefix + ' \'{}\''.format(all_sing_cmds))
    if post_commands is not None:
        command_list.extend(post_commands)
    return command_list

# if __name__ == '__main__':
#     slurm_run_scripts(header)
