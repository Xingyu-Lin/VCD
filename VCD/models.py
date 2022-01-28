import torch
import torch_scatter
from itertools import chain
from torch_geometric.nn import MetaLayer
import os


# ================== Encoder ================== #
class NodeEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, node_state):
        out = self.model(node_state)
        return out


class EdgeEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, edge_properties):
        out = self.model(edge_properties)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, node_input_size, edge_input_size, hidden_size=128, output_size=128):
        super(Encoder, self).__init__()
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.node_encoder = NodeEncoder(self.node_input_size, self.hidden_size, self.output_size)
        self.edge_encoder = EdgeEncoder(self.edge_input_size, self.hidden_size, self.output_size)

    def forward(self, node_states, edge_properties):
        node_embedding = self.node_encoder(node_states)
        edge_embedding = self.edge_encoder(edge_properties)
        return node_embedding, edge_embedding


# ================== Processor ================== #
class EdgeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(EdgeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # u_expanded = u.expand([src.size()[0], -1])
        # model_input = torch.cat([src, dest, edge_attr, u_expanded], 1)
        # out = self.model(model_input)
        model_input = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.model(model_input)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(NodeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        _, edge_dst = edge_index
        edge_attr_aggregated = torch_scatter.scatter_add(edge_attr, edge_dst, dim=0, dim_size=x.size(0))
        model_input = torch.cat([x, edge_attr_aggregated, u[batch]], dim=1)
        out = self.model(model_input)
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128):
        super(GlobalModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        node_attr_mean = torch_scatter.scatter_mean(x, batch, dim=0)
        edge_attr_mean = torch_scatter.scatter_mean(edge_attr, batch[edge_index[0]], dim=0)
        model_input = torch.cat([u, node_attr_mean, edge_attr_mean], dim=1)
        out = self.model(model_input)
        assert out.shape == u.shape
        return out


class RewardModel(torch.nn.Module):
    def __init__(self, node_size, global_size, hidden_size=128):
        super(RewardModel, self).__init__()
        self.node_size = node_size
        self.global_size = global_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.global_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, 1))

    def forward(self, node_feat, global_feat, batch):
        out = self.model(global_feat)
        return out


class GNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128):
        super(GNBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if use_global:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   GlobalModel(self.input_size[2], self.hidden_size, global_size))
        else:
            self.model = MetaLayer(EdgeModel(self.input_size[0], self.hidden_size, self.output_size),
                                   NodeModel(self.input_size[1], self.hidden_size, self.output_size),
                                   None)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        x, edge_attr, u = self.model(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u


class Processor(torch.nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=128, use_global=True, global_size=128, layers=10):
        """
        :param input_size: A list of size to edge model, node model and global model
        """
        super(Processor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_global = use_global
        self.global_size = global_size
        self.gns = torch.nn.ModuleList([
            GNBlock(self.input_size, self.hidden_size, self.output_size, self.use_global, global_size=global_size)
            for _ in range(layers)])

    def forward(self, x, edge_index, edge_attr, u, batch):
        # def forward(self, data):
        # x, edge_index, edge_attr, u, batch = data.node_embedding, data.neighbors, data.edge_embedding, data.global_feat, data.batch
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        if len(u.shape) == 1:
            u = u[None]
        if edge_index.shape[1] < 10:
            print("--------debug info---------")
            print("small number of edges")
            print("x.shape: ", x.shape)
            print("edge_index.shape: ", edge_index.shape)
            print("edge_attr.shape: ", edge_attr.shape, flush=True)
            print("--------------------------")

        x_new, edge_attr_new, u_new = x, edge_attr, u
        for gn in self.gns:
            x_res, edge_attr_res, u_res = gn(x_new, edge_index, edge_attr_new, u_new, batch)
            x_new = x_new + x_res
            edge_attr_new = edge_attr_new + edge_attr_res
            u_new = u_new + u_res
        return x_new, edge_attr_new, u_new


# ================== Decoder ================== #
class Decoder(torch.nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=3):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            # torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.output_size))

    def forward(self, node_feat, res=None):
        out = self.model(node_feat)
        if res is not None:
            out = out + res
        return out


class GNN(torch.nn.Module):
    def __init__(self, args, decoder_output_dim, name, use_reward=False):
        super(GNN, self).__init__()
        self.name = name
        self.args = args
        self.use_global = True if self.args.global_size > 1 else False
        embed_dim = 128
        self.dyn_models = torch.nn.ModuleDict({'encoder': Encoder(args.state_dim, args.relation_dim, output_size=embed_dim),
                                               'processor': Processor(
                                                   [3 * embed_dim + args.global_size,
                                                    2 * embed_dim + args.global_size,
                                                    2 * embed_dim + args.global_size],
                                                   use_global=self.use_global, layers=args.proc_layer, global_size=args.global_size),
                                               'decoder': Decoder(output_size=decoder_output_dim)})
        self.use_reward = use_reward
        print(use_reward)
        if use_reward:
            self.dyn_models['reward_model'] = RewardModel(128, 128, 128)

    def forward(self, data):
        """ data should be a dictionary containing the following dict
        edge_index: Edge index 2 x E
        x: Node feature
        edge_attr: Edge feature
        gt_accel: Acceleration label for each node
        x_batch: Batch index
        """
        out = {}
        node_embedding, edge_embedding = self.dyn_models['encoder'](data['x'], data['edge_attr'])
        n_nxt, e_nxt, lat_nxt = self.dyn_models['processor'](node_embedding,
                                                             data['edge_index'],
                                                             edge_embedding,
                                                             u=data['u'],
                                                             batch=data['x_batch'])
        # Return acceleration for each node and the final global feature (for potential multi-step training)
        if self.name == 'EdgeGNN':
            out['mesh_edge'] = self.dyn_models['decoder'](e_nxt)
        else:
            out['accel'] = self.dyn_models['decoder'](n_nxt)
            if self.use_reward:
                out['reward_nxt'] = self.dyn_models['reward_model'](n_nxt, lat_nxt, batch=data['x_batch'])

        out['n_nxt'] = n_nxt[data['partial_pc_mapped_idx']] if 'partial_pc_mapped_idx' in data else n_nxt
        out['lat_nxt'] = lat_nxt
        return out

    def load_model(self, model_path, load_names='all', load_optim=False, optim=None):
        """
        :param load_names: which part of ['encoder', 'processor', 'decoder'] to load
        :param load_optim: Whether to load optimizer states
        :return:
        """
        ckpt = torch.load(model_path)
        optim_path = model_path.replace('dyn', 'optim')
        if load_names == 'all':
            for k, v in self.dyn_models.items():
                self.dyn_models[k].load_state_dict(ckpt[k])
        else:
            for model_name in load_names:
                self.dyn_models[model_name].load_state_dict(ckpt[model_name])
        print('Loaded saved ckp from {} for {} models'.format(model_path, load_names))

        if load_optim:
            assert os.path.exists(optim_path)
            optim.load_state_dict(torch.load(optim_path))
            print('Load optimizer states from ', optim_path)

    def save_model(self, root_path, m_name, suffix, optim):
        """
        Regular saving: {input_type}_dyn_{epoch}.pth
        Best model: {input_type}_dyn_best.pth
        Optim: {input_type}_optim_{epoch}.pth
        """
        save_name = 'edge' if self.name == 'EdgeGNN' else 'dyn'
        model_path = os.path.join(root_path, '{}_{}_{}.pth'.format(m_name, save_name, suffix))
        torch.save({k: v.state_dict() for k, v in self.dyn_models.items()}, model_path)
        optim_path = os.path.join(root_path, '{}_{}_{}.pth'.format(m_name, 'optim', suffix))
        torch.save(optim.state_dict(), optim_path)

    def set_mode(self, mode='train'):
        assert mode in ['train', 'eval']
        for model in self.dyn_models.values():
            if mode == 'eval':
                model.eval()
            else:
                model.train()

    def param(self):
        model_parameters = list(chain(*[list(m.parameters()) for m in self.dyn_models.values()]))
        return model_parameters

    def to(self, device):
        for model in self.dyn_models.values():
            model.to(device)

    def freeze(self, tgts=None):
        if tgts is None:
            for m in self.dyn_models.values():
                for para in m.parameters():
                    para.requires_grad = False
        else:
            for tgt in tgts:
                m = self.dyn_models[tgt]
                for para in m.parameters():
                    para.requires_grad = False

    def unfreeze(self, tgts=None):
        if tgts is None:
            for m in self.dyn_models.values():
                for para in m.parameters():
                    para.requires_grad = True
        else:
            for tgt in tgts:
                m = self.dyn_models[tgt]
                for para in m.parameters():
                    para.requires_grad = True
