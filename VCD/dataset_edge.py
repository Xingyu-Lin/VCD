import numpy as np
import torch
from scipy import spatial
from torch_geometric.data import Data

from VCD.utils.camera_utils import get_observable_particle_index_3
from VCD.dataset import ClothDataset
from VCD.utils.utils import load_data, voxelize_pointcloud


class ClothDatasetPointCloudEdge(ClothDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        data = self._prepare_transition(idx)
        d = self.build_graph(data)
        return Data.from_dict(d)

    def _prepare_transition(self, idx, eval=False):
        pred_time_interval = self.args.pred_time_interval
        success = False
        next = 1 if not eval else self.args.time_step - self.args.n_his

        while not success:
            idx_rollout = (idx // (self.args.time_step - self.args.n_his)) % self.n_rollout
            idx_timestep = (self.args.n_his - pred_time_interval) + idx % (self.args.time_step - self.args.n_his)
            idx_timestep = max(idx_timestep, 0)

            data = load_data(self.data_dir, idx_rollout, idx_timestep, self.data_names)
            pointcloud = data['pointcloud'].astype(np.float32)
            if len(pointcloud.shape) != 2:
                print('dataset_edge.py, errorneous data. What is going on?')
                import pdb
                pdb.set_trace()

                idx += next
                continue

            if len(pointcloud) < 100:  # TODO Filter these during dataset generation
                print('dataset_edge.py, fix this')
                import pdb
                pdb.set_trace()
                idx += next
                continue

            vox_pc = voxelize_pointcloud(pointcloud, self.args.voxel_size)

            partial_particle_pos = data['positions'][data['downsample_idx']][data['downsample_observable_idx']]
            if len(vox_pc) <= len(partial_particle_pos):
                success = True

            # NOTE: what is this for?
            if eval and not success:
                return None

            idx += next

        pointcloud, partial_pc_mapped_idx = get_observable_particle_index_3(vox_pc, partial_particle_pos, threshold=self.args.voxel_size)
        normalized_vox_pc = vox_pc - np.mean(vox_pc, axis=0)

        ret_data = {
            'scene_params': data['scene_params'],
            'downsample_observable_idx': data['downsample_observable_idx'],
            'normalized_vox_pc': normalized_vox_pc,
            'partial_pc_mapped_idx': partial_pc_mapped_idx,
        }
        if eval:
            ret_data['downsample_idx'] = data['downsample_idx']
            ret_data['pointcloud'] = vox_pc

        return ret_data

    def _compute_edge_attr(self, vox_pc):
        point_tree = spatial.cKDTree(vox_pc)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = vox_pc[undirected_neighbors[0, :]] - vox_pc[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
            edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
        else:
            print("number of distance edges is 0! adding fake edges")
            edges = np.zeros((2, 2), dtype=np.uint8)
            edges[0][0] = 0
            edges[1][0] = 1
            edges[0][1] = 0
            edges[1][1] = 2
            edge_attr = np.zeros((2, self.args.relation_dim), dtype=np.float32)
            edges = torch.from_numpy(edges).bool()
            edge_attr = torch.from_numpy(edge_attr)
            print("shape of edges: ", edges.shape)
            print("shape of edge_attr: ", edge_attr.shape)

        return edges, edge_attr

    def build_graph(self, data, get_gt_edge_label=True):
        """
        data: positions, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        gt_mesh_edge: 0/1 label for groundtruth mesh edge connection.
        """
        node_attr = torch.from_numpy(data['normalized_vox_pc'])
        edges, edge_attr = self._compute_edge_attr(data['normalized_vox_pc'])

        if get_gt_edge_label:
            gt_mesh_edge = self._get_gt_mesh_edge(data, edges)
            gt_mesh_edge = torch.from_numpy(gt_mesh_edge)
        else:
            gt_mesh_edge = None

        return {
            'x': node_attr,
            'edge_index': edges,
            'edge_attr': edge_attr,
            'gt_mesh_edge': gt_mesh_edge
        }

    def _get_gt_mesh_edge(self, data, distance_edges):
        scene_params, observable_particle_idx, partial_pc_mapped_idx = data['scene_params'], data['downsample_observable_idx'], data['partial_pc_mapped_idx']
        _, cloth_xdim, cloth_ydim, _ = scene_params
        cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)

        observable_mask = np.zeros(cloth_xdim * cloth_ydim)
        observable_mask[observable_particle_idx] = 1

        num_edges = distance_edges.shape[1]
        gt_mesh_edge = np.zeros((num_edges, 1), dtype=np.float32)

        for edge_idx in range(num_edges):
            # the edge index is in the range [0, len(pointcloud) - 1]
            # needs to convert it back to the idx in the downsampled graph
            s = int(distance_edges[0][edge_idx].item())
            r = int(distance_edges[1][edge_idx].item())

            # map from pointcloud idx to observable particle index
            s = partial_pc_mapped_idx[s]
            r = partial_pc_mapped_idx[r]

            s = observable_particle_idx[s]
            r = observable_particle_idx[r]

            if (r == s + 1 or r == s - 1 or
              r == s + cloth_xdim or r == s - cloth_xdim or
              r == s + cloth_xdim + 1 or r == s + cloth_xdim - 1 or
              r == s - cloth_xdim + 1 or r == s - cloth_xdim - 1
            ):
                gt_mesh_edge[edge_idx] = 1

        return gt_mesh_edge
