from torch import nn
import torch


class E_GCL(nn.Module):
    """
    Enhanced E(n) Equivariant Convolutional Layer
    with additional normalization and improved attention
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, attention=True, normalize=True, coords_agg='mean', 
                 tanh=False, layer_norm=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.layer_norm = layer_norm
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        
        if self.layer_norm:
            self.norm = nn.LayerNorm(output_nf)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf//2),
                act_fn,
                nn.Linear(hidden_nf//2, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        if self.layer_norm:
            out = self.norm(out)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception(f'Wrong coords_agg parameter: {self.coords_agg}')
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, 
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, 
                 attention=True, normalize=True, tanh=False, 
                 input_scaling=None, coord_scale=1.0, layer_norm=True):
        '''
        Enhanced EGNN implementation with additional features:
        - Dimensionality reduction for large input features
        - Layer normalization
        - Improved attention mechanism
        - Coordinate scaling
        - Progressive layer training support

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij)
        :param input_scaling: If provided, creates a projection layer to reduce input dimensions
        :param coord_scale: Scaling factor for input coordinates
        :param layer_norm: Whether to use layer normalization
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coord_scale = coord_scale
        
        # Input dimensionality reduction if needed
        if input_scaling is not None and in_node_nf > hidden_nf:
            self.input_scaling = True
            self.embedding_in = nn.Sequential(
                nn.Linear(in_node_nf, input_scaling),
                act_fn,
                nn.Linear(input_scaling, hidden_nf)
            )
        else:
            self.input_scaling = False
            self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        
        # Create EGNN layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(
                hidden_nf, hidden_nf, hidden_nf, 
                edges_in_d=in_edge_nf,
                act_fn=act_fn, 
                residual=residual, 
                attention=attention,
                normalize=normalize, 
                tanh=tanh,
                layer_norm=layer_norm
            ))
        
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr):
        # Scale input coordinates if needed
        if self.coord_scale != 1.0:
            x = x * self.coord_scale
            
        # Embed input features
        h = self.embedding_in(h)
        
        # Process through EGNN layers
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        
        # Project to output dimension
        h = self.embedding_out(h)
        
        # Return scaled coordinates if needed
        if self.coord_scale != 1.0:
            x = x / self.coord_scale
            
        return h, x

    def load_state_dict_partial(self, state_dict, n_layers=None):
        """
        Enables progressive layer training by loading a subset of layers.
        
        :param state_dict: The state dict from a smaller model
        :param n_layers: Number of layers to load (if None, loads all available)
        """
        own_state = self.state_dict()
        
        # Load embedding layers
        for name, param in state_dict.items():
            if name in own_state and (
                'embedding_in' in name or 
                'embedding_out' in name
            ):
                own_state[name].copy_(param)
        
        # Load GCL layers up to n_layers
        max_layers = min(
            n_layers if n_layers is not None else float('inf'),
            sum(1 for name in state_dict if 'gcl_' in name) // 5  # Approximate number of parameters per layer
        )
        
        for i in range(max_layers):
            layer_name_prefix = f"gcl_{i}"
            for name, param in state_dict.items():
                if name in own_state and layer_name_prefix in name:
                    own_state[name].copy_(param)
        
        # Print information about loaded layers
        print(f"Loaded {max_layers} layers from previous model for progressive training")


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr