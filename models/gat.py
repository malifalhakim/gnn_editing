import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from .base import BaseGNNModel

from .mlp import MLP


class GAT(BaseGNNModel):
    """
    Graph Attention Network implementation.
    
    Implementation of the Graph Attention Network architecture from the paper
    "Graph Attention Networks" (Veličković et al., ICLR 2018).
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int, 
                 heads: int = 8, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False, 
                 use_linear: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize the GAT model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes/features
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            use_linear: Whether to use additional linear layers
            load_pretrained_backbone: Whether to load pretrained weights
            saved_ckpt_path: Path to the pretrained weights
        """
        super(GAT, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual, 
            use_linear=use_linear
        )
        
        # Initialize layers with attention heads
        num_heads = heads
        for i in range(num_layers):
            # Calculate input and output dimensions based on layer position
            in_dim = out_dim = hidden_channels * (1 if i == 0 else heads ** i)
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                num_heads = 1  # Use single head for output layer
            
            # Create GAT convolution layer
            conv = GATConv(in_dim, out_dim, heads=num_heads, dropout=dropout)
            self.convs.append(conv)
            
            # Add optional linear projection for residual connections
            if self.use_linear:
                self.lins.append(torch.nn.Linear(in_dim, out_dim, bias=False))


class GAT_MLP(BaseGNNModel):
    """
    Combined Graph Attention Network and MLP model.
    
    This architecture combines a GAT backbone with an MLP head,
    allowing for flexible training where components can be frozen or unfrozen.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int, 
                 heads: int = 8,
                 shared_weights: bool = True, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize the GAT_MLP model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes/features
            num_layers: Number of layers
            heads: Number of attention heads for GAT
            shared_weights: Whether to use shared weights in attention
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            load_pretrained_backbone: Whether to load pretrained GAT weights
            saved_ckpt_path: Path to the pretrained weights
        """
        super(GAT_MLP, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual
        )
        
        # Initialize GAT backbone
        if load_pretrained_backbone:
            self.GAT = GAT.from_pretrained(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                saved_ckpt_path=saved_ckpt_path,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual
            )
        else:
            self.GAT = GAT(
                in_channels=in_channels, 
                hidden_channels=hidden_channels, 
                out_channels=out_channels,
                num_layers=num_layers, 
                heads=heads, 
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
            self.freeze_layer(self.GAT, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        self.GAT.reset_parameters()
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
            train: If True, train GAT and freeze MLP
                  If False, freeze GAT and train MLP
        """
        if train:
            # Train GAT, freeze MLP
            self.freeze_layer(self.GAT, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            # Freeze GAT, train MLP
            self.freeze_layer(self.GAT, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through GAT and optionally MLP.
        
        Args:
            x: Node feature matrix
            adj_t: Adjacency matrix in sparse format
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Model output tensor
        """
        GAT_out = self.GAT(x, adj_t, *args)
        
        if self.mlp_freezed:
            # Only use GAT output if MLP is frozen
            x = GAT_out
        else:
            # Combine GAT and MLP outputs
            MLP_out = self.MLP(x, *args)
            x = GAT_out + MLP_out
            
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