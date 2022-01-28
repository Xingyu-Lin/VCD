import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric


class PrivilData(Data):
    """
    Encapsulation of multi-graphs for multi-step training
    ind: 0-(hor-1), type: vsbl or full
    Each graph contain:
        edge_index_{type}_{ind},
        x_{type}_{ind},
        edge_attr_{type}_{ind},
        gt_rwd_{type}_{ind}
        gt_accel_{type}_{ind}
        mesh_mapping_{type}_{ind}
    """

    def __init__(self, has_part=False, has_full=False, **kwargs):
        super(PrivilData, self).__init__(**kwargs)
        self.has_part = has_part
        self.has_full = has_full

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            x = key.replace('edge_index', 'x')
            return self[x].size(0)
        elif 'mesh_mapping' in key:
            # add index of mesh matching by
            x = key.replace('partial_pc_mapped_idx', 'x')
            return self[x].size(0)
        else:
            return super().__inc__(key, value)


class AggDict(dict):
    def __init__(self, is_detach=True):
        """
        Aggregate numpy arrays or pytorch tensors
        :param is_detach: Whether to save numpy arrays in stead of torch tensors
        """
        super(AggDict).__init__()
        self.is_detach = is_detach

    def __getitem__(self, item):
        return self.get(item, 0)

    def add_item(self, key, value):
        if self.is_detach and torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if not isinstance(value, torch.Tensor):
            if isinstance(value, np.ndarray) or isinstance(value, np.number):
                assert value.size == 1
            else:
                assert isinstance(value, int) or isinstance(value, float)
        if key not in self.keys():
            self[key] = value
        else:
            self[key] += value

    def update_by_add(self, src_dict):
        for key, value in src_dict.items():
            self.add_item(key, value)

    def get_mean(self, prefix, count=1):
        avg_dict = {}
        for k, v in self.items():
            avg_dict[prefix + k] = v / count
        return avg_dict


def updateDictByAdd(dict1, dict2):
    '''
    update dict1 by dict2
    '''
    for k1, v1 in dict2.items():
        for k2, v2 in v1.items():
            dict1[k1][k2] += v2.cpu().item()
    return dict1


def get_index_before_padding(graph_sizes):
    ins_len = graph_sizes.max()
    pad_len = ins_len * graph_sizes.size(0)
    valid_len = graph_sizes.sum()
    accum = torch.zeros(1).cuda()
    out = []
    for gs in graph_sizes:
        new_ind = torch.range(0, gs - 1).cuda() + accum
        out.append(new_ind)
        accum += ins_len
    final_ind = torch.cat(out, dim=0)
    return final_ind.long()


class MyDataParallel(torch_geometric.nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'module':
            return self._modules['module']
        else:
            return getattr(self.module, name)


def retrieve_data(data, key):
    """
    vsbl: [vsbl], full: [full], dual :[vsbl, full]
    """
    if isinstance(data, dict):
        identifier = '_{}'.format(key)
        out_data = {k.replace(identifier, ''): v for k, v in data.items() if identifier in k}
    return out_data
