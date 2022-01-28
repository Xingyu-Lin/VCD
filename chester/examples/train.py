import json
import os
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
)
from baselines.ddpg.main import run
from mpi4py import MPI


DEFAULT_PARAMS = {
    # env
    'env_id': 'HalfCheetah-v2',  # max absolute value of actions on different coordinates

    # ddpg
    'layer_norm': True,
    'render': False,
    'normalize_returns':False,
    'normalize_observations':True,
    'actor_lr': 0.0001,  # critic learning rate
    'critic_lr': 0.001,  # actor learning rate
    'critic_l2_reg': 1e-2,
    'popart': False,
    'gamma': 0.99,

    # training
    'seed': 0,
    'nb_epochs':500, # number of epochs
    'nb_epoch_cycles': 20,  # per epoch
    'nb_rollout_steps': 100,  # sampling batches per cycle
    'nb_train_steps': 100,  # training batches per cycle
    'batch_size': 64,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'reward_scale': 1.0,
    'clip_norm': None,

    # exploration
    'noise_type':'adaptive-param_0.2',

    # debugging, logging and visualization
    'render_eval': False,
    'nb_eval_steps':100,
    'evaluation':False,
}


def run_task(vv, log_dir=None, exp_name=None, allow_extra_parameters=False):
    # Configure logging system
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Seed for multi-CPU MPI implementation ( rank = 0 for single threaded implementation )
    rank = MPI.COMM_WORLD.Get_rank()
    rank_seed = vv['seed'] + 1000000 * rank
    set_global_seeds(rank_seed)

    # load params from config
    params = DEFAULT_PARAMS

    # update all her parameters
    if not allow_extra_parameters:
        for k,v in vv.items():
            if k not in DEFAULT_PARAMS:
                print("[ Warning ] Undefined Parameters %s with value %s"%(str(k),str(v)))
        params.update(**{k: v for (k, v) in vv.items() if k in DEFAULT_PARAMS})
    else:
        params.update(**{k: v for (k, v) in vv.items()})

    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(params, f)

    run(**params)


