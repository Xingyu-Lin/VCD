import os.path as osp
import os

# TODO change this before make it into a pip package
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = os.path.join(PROJECT_PATH, "data")

# Make sure to use absolute path
REMOTE_DIR = {
}

REMOTE_MOUNT_OPTION = {
}

REMOTE_LOG_DIR = {
}

REMOTE_HEADER = dict()

# location of the singularity file related to the project
SIMG_DIR = {
}
CUDA_MODULE = {
}
MODULES = {
}