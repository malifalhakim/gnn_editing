import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .base import BaseGNNModel
from .gcn import GCN
from .mlp import MLP


class GCN_MLP(BaseGNNModel):
    """
    Combined Graph Convolutional Network and MLP model.
    
    This model combines a GCN backbone with an MLP component,
    allowing for training where components can be selectively frozen/unfrozen.
    
    Attributes:
        GCN: Graph Convolutional Network component
        MLP: Multi-Layer Perceptron component
        mlp_freezed: Whether the MLP component is frozen
        gnn_output: Cached output from the GNN (for fast forward passes)
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int,
                 dropout: float = 0.0,
                 shared_weights: bool = True, 
                 batch_norm: bool = False, 
                 residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize GCN_MLP model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output classes/features
            num_layers: Number of layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            load_pretrained_backbone: Whether to load pretrained GCN weights
            saved_ckpt_path: Path to saved checkpoint
        """
        super(GCN_MLP, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual
        )
        
        # Initialize GCN component (backbone)
        if load_pretrained_backbone:
            self.GCN = GCN.from_pretrained(
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
            self.GCN = GCN(
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

        # Set initial frozen state
        self.mlp_freezed = True
        self.gnn_output = None
        
        # Configure which parts are trainable based on pretrained flag
        if load_pretrained_backbone:
            self.freeze_layer(self.GCN, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
        else:
            self.freeze_module(train=True)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        # Reset GCN parameters
        self.GCN.reset_parameters()
        
        # Reset MLP parameters
        self.MLP.reset_parameters()

    def freeze_layer(self, model: torch.nn.Module, freeze: bool = True) -> None:
        """
        Freeze or unfreeze parameters of a model.
        
        Args:
            model: Model to freeze/unfreeze
            freeze: Whether to freeze (True) or unfreeze (False)
        """
        for param in model.parameters():
            param.requires_grad = not freeze

    def freeze_module(self, train: bool = True) -> None:
        """
        Configure which components should be trained.
        
        Args:
            train: If True, train GCN and freeze MLP. If False, freeze GCN and train MLP.
        """
        if train:
            # Train GCN, freeze MLP
            self.freeze_layer(self.GCN, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            # Freeze GCN, train MLP
            self.freeze_layer(self.GCN, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def fast_forward(self, x: Tensor, idx: Tensor) -> Tensor:
        """
        Accelerated forward pass using cached GNN outputs.
        
        Args:
            x: Input features
            idx: Node indices to process
            
        Returns:
            Model output for the specified nodes
        """
        assert self.gnn_output is not None, "No cached GNN output available"
        assert not self.mlp_freezed, "MLP must be trainable for fast_forward"
        
        return (self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + 
                self.MLP(x))

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through both GCN and potentially MLP components.
        
        Args:
            x: Node feature matrix
            adj_t: Sparse adjacency tensor
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor with shape [num_nodes, out_channels]
        """
        # Get output from GCN component
        GCN_out = self.GCN(x, adj_t, *args)
        
        if self.mlp_freezed:
            # Only use GCN output if MLP is frozen
            x = GCN_out
        else:
            # Combine GCN and MLP outputs
            MLP_out = self.MLP(x, *args)
            x = GCN_out + MLP_out
            
        return x