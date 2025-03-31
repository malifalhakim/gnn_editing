from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, ModuleList
from torch_sparse import SparseTensor
from torch_geometric.nn import GINConv
from .base import BaseGNNModel
from .mlp import MLP


class GIN(BaseGNNModel):
    """
    Graph Isomorphism Network implementation.
    
    Implementation of the Graph Isomorphism Network (GIN) architecture from the paper
    "How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019).
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False, 
                 use_linear: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize the GIN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of GIN layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            use_linear: Whether to use additional linear layers
            load_pretrained_backbone: Whether to load pretrained weights
            saved_ckpt_path: Path to the pretrained weights
        """
        super(GIN, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual, 
            use_linear=use_linear)
        
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                num_heads = 1
            mlp = Sequential(
                Linear(in_dim, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)


    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        for idx, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            x = F.relu(batch_norm(conv(x, adj_t)))

            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear

            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]

        #x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, size):
        if self.use_linear:
            raise NotImplementedError
        
        if layer != 0:
            x = self.dropout(x)

        x_target = x[:size[1]]

        h = self.convs[layer]((x, x_target), adj_t)

        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = F.relu(h)

        return h


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i in range(len(self.convs)):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index, size).cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all



class GIN_MLP(BaseGNNModel):
    """
    Combined Graph Isomorphism Network and MLP model.
    
    This model combines a GIN backbone with an MLP component,
    allowing for editable neural networks with frozen/unfrozen components.
    
    Attributes:
        GIN: Graph Isomorphism Network component
        MLP: Multi-Layer Perceptron component
        mlp_freezed: Whether the MLP component is frozen
        gnn_output: Cached output from the GNN (for fast forward passes)
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int,
                 shared_weights: bool = True, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize GIN_MLP model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of layers
            shared_weights: Whether to use shared weights (not used in this implementation)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            load_pretrained_backbone: Whether to load pretrained GIN weights
            saved_ckpt_path: Path to saved checkpoint
        """
        super(GIN_MLP, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual
        )
        
        # Initialize GIN component (backbone)
        if load_pretrained_backbone:
            self.GIN = GIN.from_pretrained(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                saved_ckpt_path=saved_ckpt_path,
                num_layers=num_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual
            )
        else:
            self.GIN = GIN(
                in_channels=in_channels, 
                hidden_channels=hidden_channels, 
                out_channels=out_channels,
                num_layers=num_layers, 
                dropout=dropout, 
                batch_norm=batch_norm, 
                residual=residual
            )
        
        # Initialize MLP component
        self.MLP = MLP(
            in_channels=in_channels, 
            hidden_channels=hidden_channels,
            out_channels=out_channels, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_norm=batch_norm, 
            residual=residual
        )

        self.mlp_freezed = True
        if load_pretrained_backbone:
            self.freeze_layer(self.GIN, freeze=True)
            self.freeze_layer(self.GIN, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)
        self.gnn_output = None


    def reset_parameters(self):
        ### reset GIN parameters
        for conv in self.GIN.convs:
            conv.reset_parameters()
        if self.GIN.batch_norm:
            for bn in self.GIN.bns:
                bn.reset_parameters()

        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()

    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze

    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.GIN, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.GIN, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        GIN_out = self.GIN(x, adj_t, *args)
        if self.mlp_freezed:
            x = GIN_out
        else:
            MLP_out = self.MLP(x, *args)
            x = GIN_out + MLP_out
        return x

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + self.MLP(x)