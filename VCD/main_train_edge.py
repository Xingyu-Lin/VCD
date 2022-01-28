import argparse
from chester import logger
import json
import os.path as osp
from VCD.vc_edge import VCConnection
from VCD.main import create_env
from VCD.utils.utils import configure_logger, configure_seed


# TODO Merge arguments
def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/edge_debug/', help='Logging directory')
    parser.add_argument('--seed', type=int, default=100)

    # Env
    parser.add_argument('--env_name', type=str, default='ClothFlatten')
    parser.add_argument('--cached_states_path', type=str, default='1213_release_n1000.pkl')
    parser.add_argument('--num_variations', type=int, default=1000)
    parser.add_argument('--partial_observable', type=bool, default=True, help="Whether only the partial point cloud can be observed")
    parser.add_argument('--particle_radius', type=float, default=0.00625, help='Particle radius for the cloth')

    # Dataset
    parser.add_argument('--n_rollout', type=int, default=2000, help='Number of training trajectories')
    parser.add_argument('--time_step', type=int, default=100, help='Time steps per trajectory')
    parser.add_argument('--dt', type=float, default=1. / 100.)
    parser.add_argument('--pred_time_interval', type=int, default=5, help='Interval of timesteps between each dynamics prediction (model dt)')
    parser.add_argument('--train_valid_ratio', type=float, default=0.9, help="Ratio between training and validation")
    parser.add_argument('--dataf', type=str, default='softgym/softgym/cached_initial_states/', help='Path to dataset')
    parser.add_argument('--gen_data', type=int, default=0, help='Whether to generate dataset')
    parser.add_argument('--gen_gif', type=bool, default=0, help='Whether to also save gif of each trajectory (for debugging)')

    # Model
    parser.add_argument('--global_size', type=int, default=128, help="Number of hidden nodes for global in GNN")
    parser.add_argument('--n_his', type=int, default=5, help="Number of history step input to the dynamics")
    parser.add_argument('--down_sample_scale', type=int, default=3, help="Downsample the simulated cloth by a scale of 3 on each dimension")
    parser.add_argument('--voxel_size', type=float, default=0.0216)
    parser.add_argument('--neighbor_radius', type=float, default=0.045, help="Radius for connecting nearby edges")
    parser.add_argument('--collect_data_delta_move_min', type=float, default=0.15)
    parser.add_argument('--collect_data_delta_move_max', type=float, default=0.4)
    parser.add_argument('--proc_layer', type=int, default=10, help="Number of processor layers in GNN")
    parser.add_argument('--state_dim', type=int, default=3,
                        help="Dim of node feature input. Computed based on n_his: 3 x 5 + 1 dist to ground + 2 one-hot encoding of picked particle")
    parser.add_argument('--relation_dim', type=int, default=4, help="Dim of edge feature input")

    # Resume training
    parser.add_argument('--edge_model_path', type=str, default=None, help='Path to a trained edgeGNN model')
    parser.add_argument('--load_optim', type=bool, default=False, help='Load optimizer when resume training')

    # Training
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=15, help='Number of workers for dataloader')
    parser.add_argument('--use_wandb', type=bool, default=False, help='Use weight and bias for logging')
    parser.add_argument('--plot_num', type=int, default=8, help='Number of edge prediction visuals to dump per training epoch')
    parser.add_argument('--eval', type=int, default=0, help='Whether to just evaluating the model')

    args = parser.parse_args()
    return args


def main():
    args = get_default_args()
    configure_logger(args.log_dir, args.exp_name)
    configure_seed(args.seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    env = create_env(args)
    vcd_edge = VCConnection(args, env=env)
    if args.gen_data:
        vcd_edge.generate_dataset()
    else:
        vcd_edge.train()


if __name__ == '__main__':
    main()
