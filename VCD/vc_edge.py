import numpy as np
import os

from VCD.models import GNN

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os.path as osp

from chester import logger
from VCD.utils.camera_utils import get_matrix_world_to_camera, project_to_image
import matplotlib.pyplot as plt
import torch_geometric

from VCD.dataset_edge import ClothDatasetPointCloudEdge
from VCD.utils.utils import extract_numbers
from VCD.utils.data_utils import AggDict
import json
from tqdm import tqdm


class VCConnection(object):
    def __init__(self, args, env=None):
        self.args = args
        self.model = GNN(args, decoder_output_dim=1, name='EdgeGNN')  # Predict 0/1 Label for mesh edge classification
        self.device = torch.device(self.args.cuda_idx)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.param(), lr=args.lr, betas=(args.beta1, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.8, patience=3, verbose=True)
        if self.args.edge_model_path is not None:
            self.load_model(self.args.load_optim)

        self.datasets = {phase: ClothDatasetPointCloudEdge(args, 'vsbl', phase, env) for phase in ['train', 'valid']}
        follow_batch = 'x_'
        self.dataloaders = {
            x: torch_geometric.data.DataLoader(
                self.datasets[x], batch_size=args.batch_size, follow_batch=follow_batch,
                shuffle=True if x == 'train' else False, drop_last=True,
                num_workers=args.num_workers, prefetch_factor=8)
            for x in ['train', 'valid']
        }

        self.log_dir = logger.get_dir()
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
        self.load_epoch = 0

    def generate_dataset(self):
        os.system('mkdir -p ' + self.args.dataf)
        for phase in ['train', 'valid']:
            self.datasets[phase].generate_dataset()
        print('Dataset generated in', self.args.dataf)

    def plot(self, phase, epoch, i):
        data_folder = osp.join(self.args.dataf, phase)
        traj_ids = np.random.randint(0, len(os.listdir(data_folder)), self.args.plot_num)
        step_ids = np.random.randint(self.args.n_his, self.args.time_step - self.args.n_his, self.args.plot_num)
        pred_accs, pred_mesh_edges, gt_mesh_edges, edges, positionss, rgbs = [], [], [], [], [], []
        for idx, (traj_id, step_id) in enumerate(zip(traj_ids, step_ids)):
            pred_mesh_edge, gt_mesh_edge, edge, positions, rgb = self.load_data_and_predict(traj_id, step_id, self.datasets[phase])
            pred_acc = np.mean(pred_mesh_edge == gt_mesh_edge)
            pred_accs.append(pred_acc)

            if idx < 3:  # plot the first 4 edge predictions
                pred_mesh_edges.append(pred_mesh_edge)
                gt_mesh_edges.append(gt_mesh_edge)
                edges.append(edge)
                positionss.append(positions)
                rgbs.append(rgb)

        fig = plt.figure(figsize=(30, 30))
        for idx in range(min(3, len(positionss))):
            pos, edge, pred_mesh_edge, gt_mesh_edge = positionss[idx], edges[idx], pred_mesh_edges[idx], gt_mesh_edges[idx]

            predict_ax = fig.add_subplot(3, 3, idx * 3 + 1, projection='3d')
            gt_ax = fig.add_subplot(3, 3, idx * 3 + 2, projection='3d')
            both_ax = fig.add_subplot(3, 3, idx * 3 + 3, projection='3d')

            for edge_idx in range(edge.shape[1]):
                s = int(edge[0][edge_idx])
                r = int(edge[1][edge_idx])
                if pred_mesh_edge[edge_idx]:
                    predict_ax.plot([pos[s, 0], pos[r, 0]], [pos[s, 1], pos[r, 1]], [pos[s, 2], pos[r, 2]], c='r')
                    both_ax.plot([pos[s, 0], pos[r, 0]], [pos[s, 1], pos[r, 1]], [pos[s, 2], pos[r, 2]], c='r')
                if gt_mesh_edge[edge_idx]:
                    gt_ax.plot([pos[s, 0], pos[r, 0]], [pos[s, 1], pos[r, 1]], [pos[s, 2], pos[r, 2]], c='g')
                    both_ax.plot([pos[s, 0], pos[r, 0]], [pos[s, 1], pos[r, 1]], [pos[s, 2], pos[r, 2]], c='g')

            gt_ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='g', s=20)
            predict_ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', s=20)
            both_ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='g', s=20)

        plt.savefig(osp.join(self.log_dir, 'edge-prediction-{}-{}-{}.png'.format(phase, epoch, i)))
        plt.close('all')

        if rgbs[0] is not None:
            fig, axes = plt.subplots(3, 2, figsize=(30, 20))
            for idx in range(min(3, len(positionss))):
                rgb, gt_mesh_edge, pointcloud = rgbs[idx], gt_mesh_edges[idx], positionss[idx]
                pred_mesh_edge, edge = pred_mesh_edges[idx], edges[idx]

                height, width, _ = rgb.shape
                matrix_world_to_camera = get_matrix_world_to_camera(
                    self.env.camera_params[self.env.camera_name]['pos'], self.env.camera_params[self.env.camera_name]['angle']
                )
                matrix_world_to_camera = matrix_world_to_camera[:3, :]  # 3 x 4
                u, v = project_to_image(matrix_world_to_camera, pointcloud, height, width)

                predict_ax_2 = axes[idx][0]
                true_ax_2 = axes[idx][1]

                predict_ax_2.imshow(rgb)
                true_ax_2.imshow(rgb)

                for edge_idx in range(edge.shape[1]):
                    s = int(edge[0][edge_idx])
                    r = int(edge[1][edge_idx])
                    if pred_mesh_edge[edge_idx]:
                        predict_ax_2.plot([u[s], u[r]], [v[s], v[r]], c='r', linewidth=0.5)
                    if gt_mesh_edge[edge_idx]:
                        true_ax_2.plot([u[s], u[r]], [v[s], v[r]], c='r', linewidth=0.5)

                predict_ax_2.set_title("predicted edge on point cloud")
                true_ax_2.set_title("mesh edge on particles")
                predict_ax_2.scatter(u, v, c='r', s=2)
                true_ax_2.scatter(u, v, c='r', s=2)

            plt.savefig(osp.join(self.log_dir, 'edge-projected-{}-{}-{}.png'.format(phase, epoch, i)))
            plt.close('all')

        return pred_accs

    def train(self):

        # Training loop
        st_epoch = self.load_epoch
        best_valid_loss = np.inf
        for epoch in range(st_epoch, self.args.n_epoch):
            phases = ['train', 'valid'] if self.args.eval == 0 else ['valid']
            for phase in phases:
                self.set_mode(phase)
                epoch_info = AggDict(is_detach=True)

                for i, data in tqdm(enumerate(self.dataloaders[phase]), desc=f'Epoch {epoch}, phase {phase}'):
                    data = data.to(self.device).to_dict()
                    iter_info = AggDict(is_detach=False)
                    last_global = torch.zeros(self.args.batch_size, self.args.global_size, dtype=torch.float32, device=self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        data['u'] = last_global
                        pred_mesh_edge = self.model(data)
                        loss = self.bce_logit_loss(pred_mesh_edge['mesh_edge'], data['gt_mesh_edge'])  # TODO change accel to eedge
                        iter_info.add_item('loss', loss)

                    if phase == 'train':
                        self.optim.zero_grad()
                        loss.backward()
                        self.optim.step()

                    epoch_info.update_by_add(iter_info)
                    iter_info.clear()

                    epoch_len = len(self.dataloaders[phase])
                    if i == len(self.dataloaders[phase]) - 1:
                        avg_dict = epoch_info.get_mean('{}/'.format(phase), epoch_len)
                        avg_dict['lr'] = self.optim.param_groups[0]['lr']
                        for k, v in avg_dict.items():
                            logger.record_tabular(k, v)

                        pred_accs = self.plot(phase, epoch, i)

                        logger.record_tabular(phase + '/epoch', epoch)
                        logger.record_tabular(phase + '/pred_acc', np.mean(pred_accs))
                        logger.dump_tabular()

                    if phase == 'train' and i == len(self.dataloaders[phase]) - 1:
                        suffix = '{}'.format(epoch)
                        self.model.save_model(self.log_dir, 'vsbl', suffix, self.optim)

                print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
                      (phase, epoch, self.args.n_epoch, avg_dict[f'{phase}/loss'], best_valid_loss))

                if phase == 'valid':
                    cur_loss = avg_dict[f'{phase}/loss']
                    self.scheduler.step(cur_loss)
                    if (cur_loss < best_valid_loss):
                        best_valid_loss = cur_loss
                        state_dict = self.args.__dict__
                        state_dict['best_epoch'] = epoch
                        state_dict['best_valid_loss'] = cur_loss
                        with open(osp.join(self.log_dir, 'best_state.json'), 'w') as f:
                            json.dump(state_dict, f, indent=2, sort_keys=True)
                        self.model.save_model(self.log_dir, 'vsbl', 'best', self.optim)

    def load_data_and_predict(self, rollout_idx, timestep, dataset):
        args = self.args
        self.set_mode('eval')

        idx = rollout_idx * (self.args.time_step - self.args.n_his) + timestep
        data_ori = dataset._prepare_transition(idx)
        data = dataset.build_graph(data_ori)

        gt_mesh_edge = data['gt_mesh_edge'].detach().cpu().numpy()

        with torch.no_grad():
            data['x_batch'] = torch.zeros(data['x'].size(0), dtype=torch.long, device=self.device)
            data['u'] = torch.zeros([1, self.args.global_size], device=self.device)
            for key in ['x', 'edge_index', 'edge_attr']:
                data[key] = data[key].to(self.device)
            pred_mesh_edge = self.model(data)['mesh_edge']  

        pred_mesh_edge_logits = pred_mesh_edge.cpu().numpy()
        pred_mesh_edge = pred_mesh_edge_logits > 0
        edges = data['edge_index'].detach().cpu().numpy()

        return pred_mesh_edge, gt_mesh_edge, edges, data_ori['normalized_vox_pc'], None

    def infer_mesh_edges(self, args):
        """
        args: a dict
            scene_params
            pointcloud
            cuda_idx
        """
        scene_params = args['scene_params']
        point_cloud = args['pointcloud']
        cuda_idx = args.get('cuda_idx', 0)

        self.set_mode('eval')
        if cuda_idx >= 0:
            self.to(cuda_idx)
        edge_dataset = self.datasets['train']

        normalized_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        data_ori = {
            'scene_params': scene_params,
            'observable_idx': None,
            'normalized_vox_pc': normalized_point_cloud,
            'pc_to_mesh_mapping': None
        }
        data = edge_dataset.build_graph(data_ori, get_gt_edge_label=False)
        with torch.no_grad():
            data['x_batch'] = torch.zeros(data['x'].size(0), dtype=torch.long, device=self.device)
            data['u'] = torch.zeros([1, self.args.global_size], device=self.device)
            for key in ['x', 'edge_index', 'edge_attr']:
                data[key] = data[key].to(self.device)
            pred_mesh_edge_logits = self.model(data)['mesh_edge']  

        pred_mesh_edge_logits = pred_mesh_edge_logits.cpu().numpy()
        pred_mesh_edge = pred_mesh_edge_logits > 0

        edges = data['edge_index'].detach().cpu().numpy()
        senders = []
        receivers = []
        num_edges = edges.shape[1]
        for e_idx in range(num_edges):
            if pred_mesh_edge[e_idx]:
                senders.append(int(edges[0][e_idx]))
                receivers.append(int(edges[1][e_idx]))

        mesh_edges = np.vstack([senders, receivers])
        return mesh_edges

    def to(self, cuda_idx):
        self.model.to(torch.device("cuda:{}".format(cuda_idx)))

    def set_mode(self, mode='train'):
        self.model.set_mode('train' if mode == 'train' else 'eval')

    def load_model(self, load_optim=False):
        self.model.load_model(self.args.edge_model_path, load_optim=load_optim, optim=self.optim)
        self.load_epoch = extract_numbers(self.args.edge_model_path)[-1]
