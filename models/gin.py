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
            use_linear=use_linear
        )
        
        # Create separate batch norm layers specific to GIN
        self.batch_norms = ModuleList()

        # Initialize GIN convolution layers
        for i in range(num_layers):
            # Calculate input and output dimensions based on layer position
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                
            # Create MLP for GIN layer
            mlp = Sequential(
                Linear(in_dim, 2 * hidden_channels),
                BatchNorm1d(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, out_dim),
            )
            
            # Create GIN convolution with learnable epsilon
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(out_dim))
            
            # Add linear projection if requested
            if self.use_linear:
                self.lins.append(Linear(in_dim, out_dim, bias=False))

        # Optional final classification layers
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm1d(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        for conv in self.convs:
            conv.reset_parameters()
            
        for bn in self.batch_norms:
            bn.reset_parameters()
            
        if self.use_linear:
            for lin in self.lins:
                lin.reset_parameters()
                
        self.lin1.reset_parameters()
        self.batch_norm1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through the GIN model.
        
        Args:
            x: Node feature tensor
            adj_t: Adjacency tensor in sparse format
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor with log-softmax probabilities
        """
        # Process through GIN layers
        for idx, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply GIN convolution and batch normalization
            h = batch_norm(conv(x, adj_t))
            h = F.relu(h)
            
            # Apply linear transformation if specified
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
                
            # Apply residual connection if dimensions match
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]

        # Final classification layers
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


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

        # Initialize state
        self.mlp_freezed = True
        self.gnn_output = None
        
        # Set initial frozen/unfrozen state
        if load_pretrained_backbone:
            self.freeze_layer(self.GIN, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        self.GIN.reset_parameters()
        self.MLP.reset_parameters()

    def freeze_layer(self, model: torch.nn.Module, freeze: bool = True) -> None:
        """
        Freeze or unfreeze a model's parameters.
        
        Args:
            model: Model to freeze or unfreeze
            freeze: If True, parameters will be frozen (requires_grad=False)
                   If False, parameters will be trainable (requires_grad=True)
        """
        for param in model.parameters():
            param.requires_grad = not freeze

    def freeze_module(self, train: bool = True) -> None:
        """
        Configure which components should be trained.
        
        Args:
            train: If True, train GIN and freeze MLP
                  If False, freeze GIN and train MLP
        """
        if train:
            # Train GIN, freeze MLP
            self.freeze_layer(self.GIN, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            # Freeze GIN, train MLP
            self.freeze_layer(self.GIN, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through GIN and optionally MLP.
        
        Args:
            x: Node feature matrix
            adj_t: Adjacency matrix in sparse format
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Model output tensor
        """
        GIN_out = self.GIN(x, adj_t, *args)
        
        if self.mlp_freezed:
            # Only use GIN output if MLP is frozen
            x = GIN_out
        else:
            # Combine GIN and MLP outputs
            MLP_out = self.MLP(x, *args)
            x = GIN_out + MLP_out
            
        return x

    def fast_forward(self, x: Tensor, idx: Tensor) -> Tensor:
        """
        Accelerated forward pass using cached GNN outputs.
        
        Args:
            x: Input features
            idx: Node indices to process
            
        Returns:
            Model output for the specified nodes
        """
        assert self.gnn_output is not None, "GNN output must be cached before calling fast_forward"
        assert not self.mlp_freezed, "MLP must be unfrozen to use fast_forward"
        
        return (self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + 
                self.MLP(x))