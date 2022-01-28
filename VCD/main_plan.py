import numpy as np
from VCD.rs_planner import RandomShootingUVPickandPlacePlanner
from chester import logger
import json
import os.path as osp

import copy
import pyflex
import pickle
import multiprocessing as mp
from VCD.utils.utils import (
    downsample, transform_info, draw_planned_actions, visualize, draw_edge,
    pc_reward_model, voxelize_pointcloud, vv_to_args, set_picker_pos, cem_make_gif, configure_seed, configure_logger
)
from VCD.utils.camera_utils import get_matrix_world_to_camera, get_world_coords
from softgym.utils.visualization import save_numpy_as_gif

from VCD.vc_dynamics import VCDynamics
from VCD.vc_edge import VCConnection
import argparse


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='release', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/plan/', help='Logging directory')
    parser.add_argument('--seed', type=int, default=100)

    # Env
    parser.add_argument('--env_name', type=str, default='ClothFlatten', help="'ClothFlatten' or 'TshirtFlatten'")
    parser.add_argument('--cloth_type', type=str, default='tshirt-small', help="For 'TshirtFlatten', what types of tshir to use")
    parser.add_argument('--cached_states_path', type=str, default='cloth_flatten_init_states_test_40_2.pkl') 
    parser.add_argument('--num_variations', type=int, default=20) 
    parser.add_argument('--camera_name', type=str, default='default_camera')
    parser.add_argument('--down_sample_scale', type=int, default=3)
    parser.add_argument('--n_his', type=int, default=5)

    # Load model
    parser.add_argument('--edge_model_path', type=str, default=None,
                        help='Path to a trained edgeGNN model')
    parser.add_argument('--partial_dyn_path', type=str, default=None,
                        help='Path to a dynamics model using partial point cloud')
    parser.add_argument('--load_optim', type=bool, default=False, help='Load optimizer when resume training')

    # Planning
    parser.add_argument('--shooting_number', type=int, default=500, help='Number of sampled pick-and-place action for random shooting')
    parser.add_argument('--delta_y', type=float, default=0.07, help='Fixed picking height for real-world experiment')
    parser.add_argument('--delta_y_range', type=list, default=[0, 0.5], help='Sample range for the pick-and-place height in simulation')
    parser.add_argument('--move_distance_range', type=list, default=[0.05, 0.2], help='Sample range for the pick-and-place distance')
    parser.add_argument('--pull_step', type=int, default=10, help='Number of steps for doing pick-and-place on the cloth')
    parser.add_argument('--wait_step', type=int, default=6, help='Number of steps for waiting the cloth to stablize after the pick-and-place')
    parser.add_argument('--num_worker', type=int, default=6, help='Number of processes to generate the sampled pick-and-place actions in parallel')
    parser.add_argument('--task', type=str, default='flatten', help="'flatten' or 'fold'")
    parser.add_argument('--pred_time_interval', type=int, default=5, help='Interval of timesteps between each dynamics prediction (model dt)')
    parser.add_argument('--configurations', type=list, default=[i for i in range(20)], help='List of configurations to run')
    parser.add_argument('--pick_and_place_num', type=int, default=10, help='Number of pick-and-place for one smoothing trajectory')

    # Other
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.0216, help='Pointcloud voxelization size')
    parser.add_argument('--sensor_noise', type=float, default=0, help='Artificial noise added to depth sensor')
    parser.add_argument('--gpu_num', type=int, default=1, help='# of GPUs to be used')

    # Ablation
    parser.add_argument('--fix_collision_edge', type=int, default=0, help="""
        for ablation that train without mesh edges, 
        if True, fix collision edges from the first time step during planning; 
        If False, recompute collision edge at each time step
    """)
    parser.add_argument('--use_collision_as_mesh_edge', type=int, default=0, help="""
        for ablation that train with mesh edges, but remove edge GNN at test time, 
        so it uses first-time step collision edges as the mesh edges
    """)

    args = parser.parse_args()
    return args


def prepare_policy():
    # move one of the picker to be under ground
    shape_states = pyflex.get_shape_states().reshape(-1, 14)
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    # move another picker to be above the cloth
    pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
    pp = np.random.randint(len(pos))
    shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
    shape_states[0, 3:6] = pos[pp] + [0., 0.06, 0.]
    pyflex.set_shape_states(shape_states.flatten())


def create_env(args):
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS

    # create env
    env_args = copy.deepcopy(env_arg_dict[args.env_name])
    env_args['render_mode'] = 'both'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 360
    env_args['camera_width'] = 360
    env_args['camera_name'] = args.camera_name
    env_args['headless'] = True
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    assert args.env_name in ['ClothFlatten', 'TshirtFlatten']
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations
    if args.env_name == 'TshirtFlatten':
        env_args['cloth_type'] = args.cloth_type

    env = SOFTGYM_ENVS[args.env_name](**env_args)
    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env = SOFTGYM_ENVS[args.env_name](**render_env_kwargs)

    return env, render_env


def load_edge_model(edge_model_path, env):
    if edge_model_path is not None:
        edge_model_dir = osp.dirname(edge_model_path)
        edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
        edge_model_vv['eval'] = 1
        edge_model_vv['n_epoch'] = 1
        edge_model_vv['edge_model_path'] = edge_model_path
        edge_model_args = vv_to_args(edge_model_vv)

        vcd_edge = VCConnection(edge_model_args, env=env)
        print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
    else:
        print("no edge GNN model is loaded")
        vcd_edge = None

    return vcd_edge


def load_dynamics_model(args, env, vcd_edge):
    model_vv_dir = osp.dirname(args.partial_dyn_path)
    model_vv = json.load(open(osp.join(model_vv_dir, 'best_state.json')))

    model_vv[
        'fix_collision_edge'] = args.fix_collision_edge  # for ablation that train without mesh edges, if True, fix collision edges from the first time step during planning; If False, recompute collision edge at each time step
    model_vv[
        'use_collision_as_mesh_edge'] = args.use_collision_as_mesh_edge  # for ablation that train with mesh edges, but remove edge GNN at test time, so it uses first-time step collision edges as the mesh edges
    model_vv['train_mode'] = 'vsbl'
    model_vv['use_wandb'] = False
    model_vv['eval'] = 1
    model_vv['load_optim'] = False
    model_vv['pred_time_interval'] = args.pred_time_interval
    model_vv['cuda_idx'] = args.cuda_idx
    model_vv['partial_dyn_path'] = args.partial_dyn_path
    args = vv_to_args(model_vv)

    vcdynamics = VCDynamics(args, vcd_edge=vcd_edge, env=env)
    return vcdynamics


def get_rgbd_and_mask(env, sensor_noise):
    rgbd = env.get_rgbd(show_picker=False)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    if sensor_noise > 0:
        non_cloth_mask = (depth <= 0)
        depth += np.random.normal(loc=0, scale=sensor_noise,
                                  size=(depth.shape[0], depth.shape[1]))
        depth[non_cloth_mask] = 0

    return depth.copy(), rgb, depth


def main(args):
    mp.set_start_method('forkserver', force=True)

    # Configure logger
    configure_logger(args.log_dir, args.exp_name)
    log_dir = logger.get_dir()
    # Configure seed
    configure_seed(args.seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # create env
    env, render_env = create_env(args)

    # load vcdynamics
    vcd_edge = load_edge_model(args.edge_model_path, env)
    vcdynamics = load_dynamics_model(args, env, vcd_edge)

    # compute camera matrix
    camera_pos, camera_angle = env.get_camera_params()
    matrix_world_to_camera = get_matrix_world_to_camera(cam_angle=camera_angle, cam_pos=camera_pos)

    # build random shooting planner
    planner = RandomShootingUVPickandPlacePlanner(
        args.shooting_number, args.delta_y, args.pull_step, args.wait_step,
        dynamics=vcdynamics,
        reward_model=pc_reward_model,
        num_worker=args.num_worker,
        move_distance_range=args.move_distance_range,
        gpu_num=args.gpu_num,
        delta_y_range=args.delta_y_range,
        image_size=(env.camera_height, env.camera_width),
        matrix_world_to_camera=matrix_world_to_camera,
        task=args.task,
    )

    initial_states, action_trajs, configs, all_infos, all_normalized_performance = [], [], [], [], []
    for episode_idx in args.configurations:
        # setup environment, ensure the same initial configuration
        env.reset(config_id=episode_idx)

        # move one picker below the ground, set another picker randomly to a picked point / above the cloth
        prepare_policy()

        config = env.get_current_config()
        if args.env_name == 'ClothFlatten':
            cloth_xdim, cloth_ydim = config['ClothSize']
        else:
            cloth_xdim = cloth_ydim = None
        config_id = env.current_config_id
        scene_params = [env.cloth_particle_radius, cloth_xdim, cloth_ydim, config_id]

        # prepare environment and do downsample
        if args.env_name == 'ClothFlatten':
            downsample_idx, downsample_x_dim, downsample_y_dim = downsample(cloth_xdim, cloth_ydim, args.down_sample_scale)
            scene_params[1] = downsample_x_dim
            scene_params[2] = downsample_y_dim
        else:
            downsample_idx = np.arange(pyflex.get_n_particles())

        initial_state = env.get_state()
        initial_states.append(initial_state), configs.append(config)

        ret, action_traj, infos, frames = 0, [], [], []
        gt_positions, gt_shape_positions, model_pred_particle_poses, model_pred_shape_poses, predicted_edges_all = [], [], [], [], []
        actual_pick_num = 0

        flex_states, start_poses, after_poses = [env.get_state()], [], []
        obses = [env.get_image(env.camera_width, env.camera_height)]
        for pick_try_idx in range(args.pick_and_place_num):
            # prepare input data for planning
            cloth_mask, rgb, depth = get_rgbd_and_mask(env, args.sensor_noise)
            world_coordinates = get_world_coords(rgb, depth, env)[:, :, :3].reshape((-1, 3))
            pointcloud = world_coordinates[depth.flatten() > 0].astype(np.float32)
            voxel_pc = voxelize_pointcloud(pointcloud, args.voxel_size)
            observable_particle_indices = np.zeros(len(voxel_pc), dtype=np.int32)
            vel_history = np.zeros((len(observable_particle_indices), args.n_his * 3), dtype=np.float32)

            # stop if the cloth is dragged out-of-view
            if len(voxel_pc) == 0:
                print("cloth dragged out of camera view!")
                break

            picker_position, picked_points = env.action_tool._get_pos()[0], [-1, -1]
            data = {
                'pointcloud': voxel_pc,
                'vel_his': vel_history,
                'picker_position': picker_position,
                'action': env.action_space.sample(),  # action will be replaced by sampled action later
                'picked_points': picked_points,
                'scene_params': scene_params,
                'partial_pc_mapped_idx': observable_particle_indices,
            }

            # do planning
            action_sequence, model_pred_particle_pos, model_pred_shape_pos, cem_info, predicted_edges \
                = planner.get_action(data, cloth_mask=cloth_mask)
            print("config {} pick idx {}".format(config_id, pick_try_idx), flush=True)

            # set picker to start pos (pick pos)
            start_pos, after_pos = cem_info['start_pos'], cem_info['after_pos']
            start_poses.append(start_pos), after_poses.append(after_pos)
            set_picker_pos(start_pos)

            # record data for plotting
            model_pred_particle_poses.append(model_pred_particle_pos)
            model_pred_shape_poses.append(model_pred_shape_pos)
            predicted_edges_all.append(predicted_edges)

            # decompose the large 5-step action to be small 1-step actions to execute 
            if args.pred_time_interval >= 2:
                action_sequence = np.zeros((50 + 30, 8))
                action_sequence[:50, 3] = 1  # first 50 steps pick the cloth
                action_sequence[:50, :3] = (after_pos - start_pos) / 50  # delta move

            # execute the planned action, i.e., move the picked particle to the place location
            gt_positions.append(np.zeros((len(action_sequence), len(downsample_idx), 3)))
            gt_shape_positions.append(np.zeros((len(action_sequence), 2, 3)))
            for t_idx, ac in enumerate(action_sequence):
                _, reward, done, info = env.step(ac, record_continuous_video=True, img_size=360)

                imgs = info['flex_env_recorded_frames']
                frames.extend(imgs)
                info.pop("flex_env_recorded_frames")

                ret += reward
                action_traj.append(ac)

                gt_positions[pick_try_idx][t_idx] = pyflex.get_positions().reshape(-1, 4)[downsample_idx, :3]
                shape_pos = pyflex.get_shape_states().reshape(-1, 14)
                for k in range(2):
                    gt_shape_positions[pick_try_idx][t_idx][k] = shape_pos[k][:3]

            actual_pick_num += 1
            infos.append(info)
            obses.append(env.get_image(env.camera_width, env.camera_height))
            flex_states.append(env.get_state())

            # early stop if the cloth is nearly smoothed
            if info['normalized_performance'] > 0.95:
                break

        # draw the planning actions & dump the data for drawing the planning actions
        draw_data = [episode_idx, flex_states, start_poses, after_poses, obses]
        draw_planned_actions(episode_idx, obses, start_poses, after_poses, matrix_world_to_camera, log_dir)
        with open(osp.join(log_dir, '{}_draw_planned_traj.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(draw_data, f)

        # make gif visuals of the model predictions and groundtruth rollouts
        for pick_try_idx in range(0, actual_pick_num):
            # subsample the actual rollout (since we decomposed the 5-step action to be 1-step actions)
            max_idx, factor = 80, 80 / len(model_pred_particle_poses[pick_try_idx])
            subsampled_gt_pos, subsampled_shape_pos = [], []
            for t in range(len(model_pred_particle_poses[pick_try_idx])):
                subsampled_gt_pos.append(gt_positions[pick_try_idx][min(int(t * factor), max_idx - 1)])
                subsampled_shape_pos.append(gt_shape_positions[pick_try_idx][min(max_idx - 1, int(t * factor))])

            frames_model = visualize(render_env, model_pred_particle_poses[pick_try_idx], model_pred_shape_poses[pick_try_idx],
                                     config_id, range(model_pred_particle_poses[pick_try_idx].shape[1]))
            frames_gt = visualize(render_env, subsampled_gt_pos, subsampled_shape_pos, config_id, downsample_idx)

            # visualize the infered edge from edge GNN
            predicted_edges = predicted_edges_all[pick_try_idx]
            frames_edge_visual = copy.deepcopy(frames_model)
            pointcloud_pos = model_pred_particle_poses[pick_try_idx]
            for t in range(len(frames_edge_visual)):
                frames_edge_visual[t] = draw_edge(frames_edge_visual[t], predicted_edges, matrix_world_to_camera[:3, :],
                                                  pointcloud_pos[t], env.camera_height, env.camera_width)

            combined_frames = [np.hstack([frame_gt, frame_model, frame_edge]) for (frame_gt, frame_model, frame_edge) in
                               zip(frames_gt, frames_model, frames_edge_visual)]
            save_numpy_as_gif(np.array(combined_frames), osp.join(log_dir, '{}-{}.gif'.format(episode_idx, pick_try_idx)))

        # dump traj information
        normalized_performance_traj = [info['normalized_performance'] for info in infos]
        with open(osp.join(log_dir, 'normalized_performance_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(normalized_performance_traj, f)

        # logging traj information
        transformed_info = transform_info([infos])
        with open(osp.join(log_dir, 'transformed_info_traj_{}.pkl'.format(episode_idx)), 'wb') as f:
            pickle.dump(transformed_info, f)
        for info_name in transformed_info:
            final_5 = transformed_info[info_name][0, 4] if transformed_info[info_name].shape[1] > 4 \
                else transformed_info[info_name][0, -1]
            logger.record_tabular('info_' + 'final_' + info_name, transformed_info[info_name][0, -1])
            logger.record_tabular('info_' + 'final_5picks_' + info_name, final_5)
            logger.record_tabular('info_' + 'avarage_' + info_name, np.mean(transformed_info[info_name][0, :]))
            logger.record_tabular('info_' + 'sum_' + info_name, np.sum(transformed_info[info_name][0, :], axis=-1))
        logger.dump_tabular()

        cem_make_gif([frames], logger.get_dir(), args.env_name + '{}.gif'.format(episode_idx))
        all_normalized_performance.append(normalized_performance_traj)
        action_trajs.append(action_traj)
        all_infos.append(infos)

    # dump all data for reproducing the planned trajectory
    with open(osp.join(log_dir, 'all_normalized_performance.pkl'), 'wb') as f:
        pickle.dump(all_normalized_performance, f)
    traj_dict = {
        'initial_states': initial_states,
        'action_trajs': action_trajs,
        'configs': configs,
    }
    with open(osp.join(log_dir, 'cem_traj.pkl'), 'wb') as f:
        pickle.dump(traj_dict, f)


if __name__ == '__main__':
    main(get_default_args())
