import os
import os.path as osp
import copy
import cv2
import json
import wandb
import numpy as np
import scipy
from tqdm import tqdm
from chester import logger

import torch
import torch_geometric

from softgym.utils.visualization import save_numpy_as_gif

from VCD.models import GNN
from VCD.dataset import ClothDataset
from VCD.utils.data_utils import AggDict
from VCD.utils.utils import extract_numbers, pc_reward_model, visualize
from VCD.utils.camera_utils import get_matrix_world_to_camera, project_to_image


class VCDynamics(object):
    def __init__(self, args, env, vcd_edge=None):
        # Create Models
        self.args = args
        self.env = env
        self.train_mode = args.train_mode
        self.device = torch.device(self.args.cuda_idx)
        self.input_types = ['full', 'vsbl'] if self.train_mode == 'graph_imit' else [self.train_mode]
        self.models, self.optims, self.schedulers = {}, {}, {}
        for m in self.input_types:
            self.models[m] = GNN(args, decoder_output_dim=3, name=m, use_reward=False if self.train_mode == 'vsbl' else True)  # Predict acceleration
            lr = getattr(self.args, m + '_lr') if hasattr(self.args, m + '_lr') else self.args.lr
            self.optims[m] = torch.optim.Adam(self.models[m].param(), lr=lr, betas=(self.args.beta1, 0.999))
            self.schedulers[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optims[m], 'min', factor=0.8,
                                                                            patience=3, verbose=True)
            self.models[m].to(self.device)

        self.vcd_edge = vcd_edge
        self.load_model(self.args.load_optim)

        print("VCD dynamics models created")
        if self.train_mode == 'graph_imit' and not args.tune_teach:
            self.models[self.input_types[0]].freeze()

        # Create Dataloaders
        self.datasets = {phase: ClothDataset(args, self.input_types, phase, env) for phase in ['train', 'valid']}
        for phase in ['train', 'valid']: self.datasets[phase].vcd_edge = self.vcd_edge

        follow_batch = ['x_{}'.format(t) for t in self.input_types]
        self.dataloaders = {x: torch_geometric.data.DataLoader(
            self.datasets[x], batch_size=args.batch_size, follow_batch=follow_batch,
            shuffle=True if x == 'train' else False, drop_last=True,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=5 if args.num_workers > 0 else 2)
            for x in ['train', 'valid']}

        self.mse_loss = torch.nn.MSELoss()

        self.log_dir = logger.get_dir()
        if self.args.use_wandb and args.eval == 0:
            # To use wandb, you need to create an account and run 'wandb login'.
            wandb.init(project='VCD', name=args.exp_name, resume='allow',
                       id=None, settings=wandb.Settings(start_method='thread'))
            print('Weights & Biases is initialized with run name {}'.format(args.exp_name))
            wandb.config.update(args, allow_val_change=True)

    def retrieve_data(self, data, key):
        """ vsbl: [vsbl], full: [full], dual :[vsbl, full]  """
        identifier = '_{}'.format(key)
        out_data = {k.replace(identifier, ''): v for k, v in data.items() if identifier in k}
        return out_data

    def generate_dataset(self):
        os.system('mkdir -p ' + self.args.dataf)
        for phase in ['train', 'valid']:
            self.datasets[phase].generate_dataset()
        print('Dataset generated in', self.args.dataf)

    def resume_training(self):
        pass

    def load_model(self, load_optim=False):
        if self.train_mode == 'vsbl' and self.args.partial_dyn_path is not None:  # Resume training of partial model
            self.models['vsbl'].load_model(self.args.partial_dyn_path, load_optim=load_optim, optim=self.optims['vsbl'])
            self.load_epoch = int(extract_numbers(self.args.partial_dyn_path)[-1])

        if self.train_mode == 'full' and self.args.full_dyn_path is not None:  # Resume training of full model
            self.models['full'].load_model(self.args.full_dyn_path, load_optim=load_optim, optim=self.optims['full'])
            self.load_epoch = int(extract_numbers(self.args.full_dyn_path)[-1])

        if self.train_mode == 'graph_imit' and self.args.full_dyn_path is not None:
            # Imitating the full model using a partial model.
            # Need to first load the full model, and then copy weights to the partial model
            self.models['full'].load_model(self.args.full_dyn_path, load_optim=False)
            self.models['vsbl'].load_model(self.args.full_dyn_path, load_optim=False, load_names=self.args.copy_teach)
            self.load_epoch = 0

    def load_data_and_rollout(self, m_name, traj_id, phase):
        idx = traj_id * (self.args.time_step - self.args.n_his)
        dataset = self.datasets[phase]
        data = dataset.prepare_transition(idx, eval=True)
        data = dataset.remove_suffix(data, m_name)
        traj_id = data[
            'idx_rollout']  # This can be different from traj_id as some trajectory may not load due to filtering
        config_id = int(data['scene_params'][3])

        # load action sequences and true particle positions
        traj_particle_pos, actions, gt_rewards = [], [], []
        pred_time_interval = self.args.pred_time_interval
        for t in range(max(0, self.args.n_his - pred_time_interval), self.args.time_step - pred_time_interval,
                       pred_time_interval):
            t_data = dataset.load_rollout_data(traj_id, t)
            if m_name == 'vsbl':
                traj_particle_pos.append(t_data['positions'][t_data['downsample_idx']][data['partial_pc_mapped_idx']])
            else:
                traj_particle_pos.append(t_data['positions'][t_data['downsample_idx']])
            gt_rewards.append(t_data['gt_reward_crt'])
            actions.append(t_data['action'])
        res = self.rollout(
            dict(model_input_data=copy.deepcopy(data), actions=actions, reward_model=pc_reward_model, m_name=m_name))

        model_positions = res['model_positions']
        shape_positions = res['shape_positions']
        mesh_edges = res['mesh_edges']
        pred_rewards = res['pred_rewards']
        gt_pos_rewards = res['gt_pos_rewards']

        pos_errors = []
        reward_pred_errors = []
        for i in range(len(actions)):
            pos_error = np.mean(np.linalg.norm(model_positions[i] - traj_particle_pos[i], axis=1))
            pos_errors.append(pos_error)
            reward_pred_errors.append((pred_rewards[i] - gt_pos_rewards[i]) ** 2)
        reward_pred_error = np.mean(reward_pred_errors)  # measure only reward prediction error
        planning_error = (pred_rewards[-1] - gt_rewards[-1]) ** 2  # measure predicted return error
        return {'model_positions': model_positions,
                'gt_positions': traj_particle_pos,
                'shape_positions': shape_positions,
                'config_id': config_id,
                'mesh_edges': mesh_edges,
                'reward_pred_error': reward_pred_error,
                'planning_error': planning_error,
                'rollout_pos_error': pos_errors}

    def train(self):
        # Training loop
        st_epoch = self.load_epoch if hasattr(self, 'load_epoch') else 0
        print('st epoch ', st_epoch)
        best_valid_loss = {m_name: np.inf for m_name in self.models}
        phases = ['train', 'valid'] if self.args.eval == 0 else ['valid']
        for epoch in range(st_epoch, self.args.n_epoch):
            for phase in phases:
                self.set_mode(phase)
                # Log all the useful metrics
                epoch_infos = {m: AggDict(is_detach=True) for m in self.models}

                epoch_len = len(self.dataloaders[phase])
                for i, data in tqdm(enumerate(self.dataloaders[phase]), desc=f'Epoch {epoch}, phase {phase}'):
                    data = data.to(self.device).to_dict()
                    iter_infos = {m_name: AggDict(is_detach=False) for m_name in self.models}
                    preds = {}
                    last_global = torch.zeros(self.args.batch_size, self.args.global_size, dtype=torch.float32,
                                              device=self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        for (m_name, model), iter_info in zip(self.models.items(), iter_infos.values()):
                            inputs = self.retrieve_data(data, m_name)
                            inputs['u'] = last_global
                            pred = model(inputs)
                            preds[m_name] = pred
                            iter_info.add_item('accel_loss', self.mse_loss(pred['accel'], inputs['gt_accel']))
                            iter_info.add_item('sqrt_accel_loss', torch.sqrt(iter_info['accel_loss']))
                            if self.train_mode != 'vsbl':
                                iter_info.add_item('reward_loss',
                                                   self.mse_loss(pred['reward_nxt'].squeeze(), inputs['gt_reward_nxt']))

                    if self.args.train_mode == 'graph_imit':  # Graph imitation
                        iter_infos['vsbl'].add_item('imit_node_loss', self.mse_loss(preds['vsbl']['n_nxt'],
                                                                                    preds['full']['n_nxt'].detach()))
                        iter_infos['vsbl'].add_item('imit_lat_loss', self.mse_loss(preds['vsbl']['lat_nxt'],
                                                                                   preds['full']['lat_nxt'].detach()))
                    for m_name in self.models:
                        iter_info = iter_infos[m_name]
                        for feat in ['n_nxt', 'lat_nxt']:  # Node and global output
                            iter_info.add_item(feat + '_norm', torch.norm(preds[m_name][feat], dim=1).mean())

                        if self.args.train_mode == 'vsbl':  # Student loss
                            iter_info.add_item('total_loss', iter_info['accel_loss'])
                        elif self.args.train_mode == 'graph_imit' and m_name == 'vsbl':  # Student loss
                            iter_info.add_item('imit_loss',
                                               iter_info['imit_lat_loss'] * self.args.imit_w_lat + iter_info[
                                                   'imit_node_loss'])
                            iter_info.add_item('total_loss',
                                               iter_info['accel_loss'] + self.args.imit_w * iter_info[
                                                   'imit_loss'] +
                                               + self.args.reward_w * iter_info['reward_loss'])
                        else:  # Teacher loss or no graph imitation
                            iter_info.add_item('total_loss',
                                               iter_info['accel_loss'] + self.args.reward_w * iter_info['reward_loss'])

                        if phase == 'train':
                            if not (self.train_mode == 'graph_imit' and m_name == 'full' and not self.args.tune_teach):
                                self.optims[m_name].zero_grad()
                                iter_info['total_loss'].backward()
                                self.optims[m_name].step()

                        epoch_infos[m_name].update_by_add(iter_infos[m_name])  # Aggregate info

                # rollout evaluation
                nstep_eval_rollout = self.args.nstep_eval_rollout
                data_folder = osp.join(self.args.dataf, phase)
                traj_ids = np.random.permutation(len(os.listdir(data_folder)))[:nstep_eval_rollout]
                rollout_infos = {}
                for m_name in self.models:
                    rollout_info = AggDict()
                    for idx, traj_id in enumerate(traj_ids):
                        with torch.no_grad():
                            self.set_mode('eval')
                            traj_rollout_info = self.load_data_and_rollout(m_name, traj_id, phase)

                        rollout_info.update_by_add(
                            dict(rollout_pos_error=np.array(traj_rollout_info['rollout_pos_error']).mean(),
                                 reward_pred_error=np.array(traj_rollout_info['reward_pred_error']).mean(),
                                 planning_error=np.array(traj_rollout_info['planning_error']).mean()))

                        frames_model = visualize(self.datasets[phase].env, traj_rollout_info['model_positions'],
                                                 traj_rollout_info['shape_positions'],
                                                 traj_rollout_info['config_id'])
                        frames_gt = visualize(self.datasets[phase].env, traj_rollout_info['gt_positions'],
                                              traj_rollout_info['shape_positions'],
                                              traj_rollout_info['config_id'])
                        mesh_edges = traj_rollout_info['mesh_edges']
                        if mesh_edges is not None:  # Visualization of mesh edges on the predicted model
                            frames_edge_visual = copy.deepcopy(frames_model)
                            matrix_world_to_camera = get_matrix_world_to_camera()[:3, :]  # 3 x 4
                            for t in range(len(frames_edge_visual)):
                                u, v = project_to_image(matrix_world_to_camera, traj_rollout_info['model_positions'][t])
                                for edge_idx in range(mesh_edges.shape[1]):
                                    s = mesh_edges[0][edge_idx]
                                    r = mesh_edges[1][edge_idx]
                                    start = (u[s], v[s])
                                    end = (u[r], v[r])
                                    color = (255, 0, 0)
                                    thickness = 1
                                    image = cv2.line(frames_edge_visual[t], start, end, color, thickness)
                                    frames_edge_visual[t] = image

                            combined_frames = [np.hstack([frame_gt, frame_model, frame_edge])
                                               for (frame_gt, frame_model, frame_edge) in
                                               zip(frames_gt, frames_model, frames_edge_visual)]

                        else:
                            combined_frames = [np.hstack([frame_gt, frame_model]) for (frame_gt, frame_model) in
                                               zip(frames_gt, frames_model)]
                        if idx < 5:
                            save_numpy_as_gif(np.array(combined_frames),
                                              osp.join(self.log_dir,
                                                       '{}-{}-{}-{}.gif'.format(m_name, phase, epoch, idx)))
                    rollout_infos[m_name] = rollout_info.get_mean(f"{m_name}/{phase}/", len(traj_ids))

                if phase == 'train' and epoch % self.args.save_model_interval == 0:
                    for m_name, model in self.models.items():
                        suffix = '{}'.format(epoch)
                        model.save_model(self.log_dir, m_name, suffix, self.optims[m_name])

                if phase == 'valid':
                    for m_name, model in self.models.items():
                        epoch_info = epoch_infos[m_name]
                        cur_loss = epoch_info[f"{m_name}/{phase}/" + 'total_loss']
                        if not self.args.fixed_lr:
                            self.schedulers[m_name].step(cur_loss)
                        if cur_loss < best_valid_loss[m_name]:
                            best_valid_loss[m_name] = cur_loss
                            state_dict = self.args.__dict__
                            state_dict['best_epoch'] = epoch
                            state_dict['best_valid_loss'] = cur_loss
                            with open(osp.join(self.log_dir, 'best_state.json'), 'w') as f:
                                json.dump(state_dict, f, indent=2, sort_keys=True)
                            model.save_model(self.log_dir, m_name, 'best', self.optims[m_name])
                # logging
                logger.record_tabular(phase + '/epoch', epoch)
                for m_name in self.models:
                    epoch_info, rollout_info = epoch_infos[m_name], rollout_infos[m_name]
                    epoch_info = epoch_info.get_mean(f"{m_name}/{phase}/", epoch_len)
                    epoch_info['lr'] = self.optims[m_name].param_groups[0]['lr']
                    logger.log(
                        f'{phase} [{epoch}/{self.args.n_epoch}] Loss: {epoch_info[f"{m_name}/{phase}/total_loss"]:.4f}',
                        best_valid_loss[m_name])

                    for k, v in epoch_info.items():
                        logger.record_tabular(k, v)
                    for k, v in rollout_info.items():
                        logger.record_tabular(k, v)

                    if self.args.use_wandb and self.args.eval == 0:
                        wandb.log(epoch_info, step=epoch)
                        wandb.log(rollout_info, step=epoch)

                logger.dump_tabular()

    def set_mode(self, mode='train'):
        for model in self.models.values():
            model.set_mode('train' if mode == 'train' else 'eval')

    def to(self, cuda_idx):
        for model in self.models.values():
            model.to(torch.device("cuda:{}".format(cuda_idx)))

    def rollout(self, args):
        """
        args need to contain the following contents:
            model_input_data: current point cloud, velocity history, picked point, picker position, etc
            actions: rollout actions
            reward_model: reward function
            cuda_idx (optional): default 0
            robot_exp (optional): default False

        return a dict:
            final_ret: final reward of the rollout
            model_positions: model predicted point cloud positions
            shape_positions: positions of the pickers, for visualization
            mesh_edges: predicted mesh edge
            time_cost: time cost for different parts of the rollout function
        """
        model_input_data = args['model_input_data']
        actions = args['actions']  # NOTE: sequence of actions to rollout
        reward_model = args['reward_model']
        m_name = args['m_name']
        dataset = self.datasets['train']  # Both train and valid are the same during inference
        H = len(actions)  # Planning horizon
        cuda_idx = args.get('cuda_idx', 0)
        robot_exp = args.get('robot_exp', False)

        self.set_mode('eval')
        self.to(cuda_idx)
        self.device = torch.device(cuda_idx)

        pc_pos = model_input_data['pointcloud']
        pc_vel_his = model_input_data['vel_his']
        picker_pos = model_input_data['picker_position']
        # picked_particles = model_input_data['picked_points']
        scene_params = model_input_data['scene_params']
        observable_particle_index = model_input_data['partial_pc_mapped_idx']
        rest_dist = model_input_data.get('rest_dist', None)
        mesh_edges = model_input_data.get('mesh_edges', None)
        assert rest_dist is None  # The rest_dist will be computed from the initial_particle_pos?

        # record model predicted point cloud positions
        model_positions = np.zeros((H, len(pc_pos), 3))
        shape_positions = np.zeros((H, 2, 3))
        initial_pc_pos = pc_pos.copy()
        pred_rewards = np.zeros(H)
        gt_pos_rewards = np.zeros(H)

        # Predict mesh during evaluation and use gt edges during training
        if self.vcd_edge is not None and mesh_edges is None:
            model_input_data['cuda_idx'] = cuda_idx
            mesh_edges = self.vcd_edge.infer_mesh_edges(model_input_data)

        # for ablation that uses first-time step collision edges as the mesh edges
        if self.datasets['train'].args.use_collision_as_mesh_edge:
            print("construct collision edges at the first time step as mesh edges!")
            neighbor_radius = self.datasets['train'].args.neighbor_radius
            point_tree = scipy.spatial.cKDTree(pc_pos)
            undirected_neighbors = np.array(list(point_tree.query_pairs(neighbor_radius, p=2))).T
            mesh_edges = np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1)

        ret = 0
        for t in range(H):
            data = {'pointcloud': pc_pos,
                    'vel_his': pc_vel_his,
                    'picker_position': picker_pos,
                    'action': actions[t],
                    # 'picked_points': picked_particles,
                    'scene_params': scene_params,
                    'partial_pc_mapped_idx': observable_particle_index if not robot_exp else range(len(pc_pos)),
                    'mesh_edges': mesh_edges,
                    'rest_dist': rest_dist,
                    'initial_particle_pos': initial_pc_pos}

            # for ablation that fixes the collision edge as those computed in the firs time step
            if not self.datasets['train'].args.fix_collision_edge:
                graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
            else:
                logger.log('using fixed collision edge!')
                if t == 0:  # store the initial collision edge
                    graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
                    fix_edges, fix_edge_attr = graph_data['neighbors'], graph_data['edge_attr']
                else:  # use the stored initial edge
                    graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
                    graph_data['neighbors'], graph_data['edge_attr'] = fix_edges, fix_edge_attr

            model_positions[t] = pc_pos
            shape_positions[t] = picker_pos

            inputs = {'x': graph_data['node_attr'].to(self.device),
                      'edge_attr': graph_data['edge_attr'].to(self.device),
                      'edge_index': graph_data['neighbors'].to(self.device),
                      'x_batch': torch.zeros(graph_data['node_attr'].size(0), dtype=torch.long, device=self.device),
                      'u': torch.zeros([1, self.args.global_size], device=self.device)}

            # obtain model predictions
            with torch.no_grad():
                pred = self.models[m_name](inputs)
                pred_accel = pred['accel'].cpu().numpy()
                pred_reward = pred['reward_nxt'].cpu().numpy() if 'reward_nxt' in pred else 0.

            pc_pos, pc_vel_his, picker_pos = self.update_graph(pred_accel, pc_pos, pc_vel_his,
                                                               graph_data['picked_status'],
                                                               graph_data['picked_particles'])
            reward = reward_model(pc_pos)
            ret += reward

            pred_rewards[t] = pred_reward
            gt_pos_rewards[t] = reward

            if t == H - 1:
                final_ret = reward
        if mesh_edges is None:  # No mesh edges input during training
            mesh_edges = data['mesh_edges']  # This is modified inside prepare_transition function
        return dict(final_ret=final_ret,
                    model_positions=model_positions,
                    shape_positions=shape_positions,
                    mesh_edges=mesh_edges,
                    pred_rewards=pred_rewards,
                    gt_pos_rewards=gt_pos_rewards)

    def update_graph(self, pred_accel, pc_pos, velocity_his, picked_status, picked_particles):
        """ Euler integration"""
        # vel_his: [v(t-20), ... v(t)], v(t) = (p(t) - o(t-5)) / (5*dt)
        pred_time_interval = self.args.pred_time_interval
        pred_vel = velocity_his[:, -3:] + pred_accel * self.args.dt * pred_time_interval
        pc_pos = pc_pos + pred_vel * self.args.dt * pred_time_interval

        # udpate position and velocity from the model prediction
        velocity_his = np.hstack([velocity_his[:, 3:], pred_vel])

        # the picked particles position and velocity should remain the same as before
        cnt = 0
        picked_vel, picked_pos, new_picker_pos = picked_status
        for p_idx in picked_particles:
            if p_idx != -1:
                pc_pos[p_idx] = picked_pos[cnt]
                velocity_his[p_idx] = picked_vel[cnt]
                cnt += 1

        # update picker position, and the particles picked
        picker_pos = new_picker_pos
        return pc_pos, velocity_his, picker_pos
