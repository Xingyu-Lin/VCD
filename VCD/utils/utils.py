import os.path as osp
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from VCD.utils.camera_utils import project_to_image
import pyflex
import re
import h5py
import os
from softgym.utils.visualization import save_numpy_as_gif
from chester import logger
import random


class VArgs(object):
    def __init__(self, vv):
        for key, val in vv.items():
            setattr(self, key, val)


def vv_to_args(vv):
    args = VArgs(vv)
    return args


# Function to extract all the numbers from the given string
def extract_numbers(str):
    array = re.findall(r'[0-9]+', str)
    if len(array) == 0:
        return [0]
    return array


################## Pointcloud Processing #################
import pcl


# def get_partial_particle(full_particle, observable_idx):
#     return np.array(full_particle[observable_idx], dtype=np.float32)


def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = pcl.PointCloud(pointcloud)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    pointcloud = sor.filter()
    pointcloud = np.asarray(pointcloud).astype(np.float32)
    return pointcloud


from softgym.utils.misc import vectorized_range, vectorized_meshgrid


def pc_reward_model(pos, cloth_particle_radius=0.00625, downsample_scale=3):
    cloth_particle_radius *= downsample_scale
    pos = np.reshape(pos, [-1, 3])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 2])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 2])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 2]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)

    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    res = np.sum(grid) * span[0] * span[1]
    return res


################## IO #################################
def downsample(cloth_xdim, cloth_ydim, scale):
    cloth_xdim, cloth_ydim = int(cloth_xdim), int(cloth_ydim)
    new_idx = np.arange(cloth_xdim * cloth_ydim).reshape((cloth_ydim, cloth_xdim))
    new_idx = new_idx[::scale, ::scale]
    cloth_ydim, cloth_xdim = new_idx.shape
    new_idx = new_idx.flatten()

    return new_idx, cloth_xdim, cloth_ydim


def load_h5_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = {}
    for name in data_names:
        d = np.array(hf.get(name))
        data[name] = d
    hf.close()
    return data


def store_h5_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for name in data_names:
        hf.create_dataset(name, data=data[name])
    hf.close()


def load_data(data_dir, idx_rollout, idx_timestep, data_names):
    data_path = os.path.join(data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
    return load_h5_data(data_names, data_path)


def load_data_list(data_dir, idx_rollout, idx_timestep, data_names):
    data_path = os.path.join(data_dir, str(idx_rollout), str(idx_timestep) + '.h5')
    d = load_h5_data(data_names, data_path)
    return [d[name] for name in data_names]


def store_data():
    raise NotImplementedError


def transform_info(all_infos):
    """ Input: All info is a nested list with the index of [episode][time]{info_key:info_value}
        Output: transformed_infos is a dictionary with the index of [info_key][episode][time]
    """
    if len(all_infos) == 0:
        return []
    transformed_info = {}
    num_episode = len(all_infos)
    T = len(all_infos[0])

    for info_name in all_infos[0][0].keys():
        infos = np.zeros([num_episode, T], dtype=np.float32)
        for i in range(num_episode):
            infos[i, :] = np.array([info[info_name] for info in all_infos[i]])
        transformed_info[info_name] = infos
    return transformed_info


def draw_grid(list_of_imgs, nrow, padding=10, pad_value=200):
    img_list = torch.from_numpy(np.array(list_of_imgs).transpose(0, 3, 1, 2))
    img = make_grid(img_list, nrow=nrow, padding=padding, pad_value=pad_value)
    # print(img.shape)
    img = img.numpy().transpose(1, 2, 0)
    return img


def inrange(x, low, high):
    if x >= low and x < high:
        return True
    else:
        return False


################## Visualization ######################

def draw_edge(frame, predicted_edges, matrix_world_to_camera, pointcloud, camera_height, camera_width):
    u, v = project_to_image(matrix_world_to_camera, pointcloud, camera_height, camera_width)
    for edge_idx in range(predicted_edges.shape[1]):
        s = predicted_edges[0][edge_idx]
        r = predicted_edges[1][edge_idx]
        start = (u[s], v[s])
        end = (u[r], v[r])
        color = (255, 0, 0)
        thickness = 1
        image = cv2.line(frame, start, end, color, thickness)

    return image


def cem_make_gif(all_frames, save_dir, save_name):
    # Convert to T x index x C x H x W for pytorch
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [make_grid(torch.from_numpy(frame), nrow=5).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]
    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


def draw_policy_action(obs_before, obs_after, start_loc_1, end_loc_1, matrix_world_to_camera, start_loc_2=None, end_loc_2=None):
    height, width, _ = obs_before.shape
    if start_loc_2 is not None:
        l = [(start_loc_1, end_loc_1), (start_loc_2, end_loc_2)]
    else:
        l = [(start_loc_1, end_loc_1)]
    for (start_loc, end_loc) in l:
        # print(start_loc, end_loc)
        suv = project_to_image(matrix_world_to_camera, start_loc.reshape((1, 3)), height, width)
        su, sv = suv[0][0], suv[1][0]
        euv = project_to_image(matrix_world_to_camera, end_loc.reshape((1, 3)), height, width)
        eu, ev = euv[0][0], euv[1][0]
        if inrange(su, 0, width) and inrange(sv, 0, height) and inrange(eu, 0, width) and inrange(ev, 0, height):
            cv2.arrowedLine(obs_before, (su, sv), (eu, ev), (255, 0, 0), 3)
            obs_before[sv - 5:sv + 5, su - 5:su + 5, :] = (0, 0, 0)

    res = np.concatenate((obs_before, obs_after), axis=1)
    return res


def draw_planned_actions(save_idx, obses, start_poses, end_poses, matrix_world_to_camera, log_dir):
    height = width = obses[0].shape[0]

    start_uv = []
    end_uv = []
    for sp in start_poses:
        suv = project_to_image(matrix_world_to_camera, sp.reshape((1, 3)), height, width)
        start_uv.append((suv[0][0], suv[1][0]))
    for ep in end_poses:
        euv = project_to_image(matrix_world_to_camera, ep.reshape((1, 3)), height, width)
        end_uv.append((euv[0][0], euv[1][0]))

    res = []
    for idx in range(len(obses) - 1):
        obs = obses[idx]
        su, sv = start_uv[idx]
        eu, ev = end_uv[idx]
        if inrange(su, 0, width) and inrange(sv, 0, height) and inrange(eu, 0, width) and inrange(ev, 0, height):
            cv2.arrowedLine(obs, (su, sv), (eu, ev), (255, 0, 0), 3)
            obs[sv - 5:sv + 5, su - 5:su + 5, :] = (0, 0, 0)
        res.append(obs)

    res.append(obses[-1])
    res = np.concatenate(res, axis=1)
    cv2.imwrite(osp.join(log_dir, '{}_planned.png'.format(save_idx)), res[:, :, ::-1])


def draw_cem_elites(obs_, start_poses, end_poses, mean_start_pos, mean_end_pos,
                    matrix_world_to_camera, log_dir, save_idx=None):
    obs = obs_.copy()
    start_uv = []
    end_uv = []
    height = width = obs.shape[0]
    for sp in start_poses:
        suv = project_to_image(matrix_world_to_camera, sp.reshape((1, 3)), height, width)
        start_uv.append((suv[0][0], suv[1][0]))
    for ep in end_poses:
        euv = project_to_image(matrix_world_to_camera, ep.reshape((1, 3)), height, width)
        end_uv.append((euv[0][0], euv[1][0]))

    for idx in range(len(start_poses)):
        su, sv = start_uv[idx]
        eu, ev = end_uv[idx]
        # poses at the front have higher reward
        if inrange(su, 0, 255) and inrange(sv, 0, 255) and inrange(eu, 0, 255) and inrange(ev, 0, 255):
            cv2.arrowedLine(obs, (su, sv), (eu, ev), (255 * (1 - idx / len(start_poses)), 0, 0), 2)
            obs[sv - 2:sv + 2, su - 2:su + 2, :] = (0, 0, 0)

    mean_s_uv = project_to_image(matrix_world_to_camera, mean_start_pos.reshape((1, 3)), height, width)
    mean_e_uv = project_to_image(matrix_world_to_camera, mean_end_pos.reshape((1, 3)), height, width)
    mean_su, mean_sv = mean_s_uv[0][0], mean_s_uv[1][0]
    mean_eu, mean_ev = mean_e_uv[0][0], mean_e_uv[1][0]

    if inrange(mean_su, 0, 255) and inrange(mean_sv, 0, 255) and \
      inrange(mean_eu, 0, 255) and inrange(mean_ev, 0, 255):
        cv2.arrowedLine(obs, (mean_su, mean_sv), (mean_eu, mean_ev), (0, 0, 255), 3)
        obs[mean_su - 5:mean_sv + 5, mean_eu - 5:mean_ev + 5, :] = (0, 0, 0)
    if save_idx is not None:
        cv2.imwrite(osp.join(log_dir, '{}_elite.png'.format(save_idx)), obs)
    return obs


def set_shape_pos(pos):
    shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
    shape_states[:, 3:6] = pos.reshape(-1, 3)
    shape_states[:, :3] = pos.reshape(-1, 3)
    pyflex.set_shape_states(shape_states)


def visualize(env, particle_positions, shape_positions, config_id, sample_idx=None, picked_particles=None, show=False):
    """ Render point cloud trajectory without running the simulation dynamics"""
    env.reset(config_id=config_id)
    frames = []
    for i in range(len(particle_positions)):
        particle_pos = particle_positions[i]
        shape_pos = shape_positions[i]
        p = pyflex.get_positions().reshape(-1, 4)
        p[:, :3] = [0., -0.1, 0.]  # All particles moved underground
        if sample_idx is None:
            p[:len(particle_pos), :3] = particle_pos
        else:
            p[:, :3] = [0, -0.1, 0]
            p[sample_idx, :3] = particle_pos
        pyflex.set_positions(p)
        set_shape_pos(shape_pos)
        rgb = env.get_image(env.camera_width, env.camera_height)
        frames.append(rgb)
        if show:
            if i == 0: continue
            picked_point = picked_particles[i]
            phases = np.zeros(pyflex.get_n_particles())
            for id in picked_point:
                if id != -1:
                    phases[sample_idx[int(id)]] = 1
            pyflex.set_phases(phases)
            img = env.get_image()

            cv2.imshow('picked particle images', img[:, :, ::-1])
            cv2.waitKey()

    return frames


def add_occluded_particles(observable_positions, observable_vel_history, particle_radius=0.00625, neighbor_distance=0.0216):
    occluded_idx = np.where(observable_positions[:, 1] > neighbor_distance / 2 + particle_radius)
    occluded_positions = []
    for o_idx in occluded_idx[0]:
        pos = observable_positions[o_idx]
        occlude_num = np.floor(pos[1] / neighbor_distance).astype('int')
        for i in range(occlude_num):
            occluded_positions.append([pos[0], particle_radius + i * neighbor_distance, pos[2]])

    print("add occluded particles num: ", len(occluded_positions))
    occluded_positions = np.asarray(occluded_positions, dtype=np.float32).reshape((-1, 3))
    occluded_velocity_his = np.zeros((len(occluded_positions), observable_vel_history.shape[1]), dtype=np.float32)

    all_positions = np.concatenate([observable_positions, occluded_positions], axis=0)
    all_vel_his = np.concatenate([observable_vel_history, occluded_velocity_his], axis=0)
    return all_positions, all_vel_his


def sort_pointcloud_for_fold(pointcloud, dim):
    pointcloud = list(pointcloud)
    sorted_pointcloud = sorted(pointcloud, key=lambda k: (k[0], k[2]))
    for idx in range(len(sorted_pointcloud) - 1):
        assert sorted_pointcloud[idx][0] < sorted_pointcloud[idx + 1][0] or (
          sorted_pointcloud[idx][0] == sorted_pointcloud[idx + 1][0] and
          sorted_pointcloud[idx][2] < sorted_pointcloud[idx + 1][2]
        )

    real_sorted = []
    for i in range(dim):
        points_row = sorted_pointcloud[i * dim: (i + 1) * dim]
        points_row = sorted(points_row, key=lambda k: k[2])
        real_sorted += points_row

    sorted_pointcloud = real_sorted

    return np.asarray(sorted_pointcloud)


def get_fold_idx(dim=4):
    group_a = []
    for i in range(dim - 1):
        for j in range(dim - i - 1):
            group_a.append(i * dim + j)

    group_b = []
    for j in range(dim - 1, 0, -1):
        for i in range(dim - 1, dim - 1 - j, -1):
            group_b.append(i * dim + j)

    return group_a, group_b


############################ Other ########################
def updateDictByAdd(dict1, dict2):
    '''
    update dict1 by dict2
    '''
    for k1, v1 in dict2.items():
        for k2, v2 in v1.items():
            dict1[k1][k2] += v2.cpu().item()
    return dict1


def configure_logger(log_dir, exp_name):
    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)


def configure_seed(seed):
    # Configure seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


############### for planning ###############################
def set_picker_pos(pos):
    shape_states = pyflex.get_shape_states().reshape((-1, 14))
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    shape_states[0, :3] = pos
    shape_states[0, 3:6] = pos
    pyflex.set_shape_states(shape_states)
    pyflex.step()


def set_resource():
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
