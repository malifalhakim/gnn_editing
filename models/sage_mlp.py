import torch

from torch import Tensor
from torch_sparse import SparseTensor
from .base import BaseGNNModel

from .sage import SAGE
from .mlp import MLP


class SAGE_MLP(BaseGNNModel):
    """
    Combined GraphSAGE and MLP model.
    
    This model combines a GraphSAGE backbone with an MLP component,
    allowing for editable neural networks with frozen/unfrozen components.
    
    Attributes:
        SAGE: GraphSAGE component
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
        Initialize SAGE_MLP model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of layers
            shared_weights: Whether to use shared weights (not used in this implementation)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            load_pretrained_backbone: Whether to load pretrained SAGE weights
            saved_ckpt_path: Path to saved checkpoint
        """
        super(SAGE_MLP, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual
        )
        
        # Initialize SAGE component (backbone)
        if load_pretrained_backbone:
            self.SAGE = SAGE.from_pretrained(
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
            self.SAGE = SAGE(
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
            residual=residual,
            layer_norm=True,              # Enable layer normalization
            spectral_norm_enabled=True,   # Enable spectral normalization
            activity_regularization=1e-4, # Add L2 penalty on activations
            dropout_type='concrete',      # Use concrete dropout instead of standard
            weight_decay=1e-5          # L2 regularization strength (used during optimization)
        )

        # Set initial frozen state
        self.mlp_freezed = True
        self.gnn_output = None
        
        # Configure which parts are trainable based on pretrained flag
        if load_pretrained_backbone:
            self.freeze_layer(self.SAGE, freeze=True)
            self.freeze_layer(self.MLP, freeze=True)
        else:
            self.freeze_module(train=True)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        # Reset SAGE parameters
        self.SAGE.reset_parameters()
        
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
            train: If True, train SAGE and freeze MLP. If False, freeze SAGE and train MLP.
        """
        if train:
            # Train SAGE, freeze MLP
            self.freeze_layer(self.SAGE, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            # Freeze SAGE, train MLP
            self.freeze_layer(self.SAGE, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through both SAGE and potentially MLP components.
        
        Args:
            x: Node feature matrix
            adj_t: Sparse adjacency tensor
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor with shape [num_nodes, out_channels]
        """
        # Get output from SAGE component
        SAGE_out = self.SAGE(x, adj_t, *args)
        
        if self.mlp_freezed:
            # Only use SAGE output if MLP is frozen
            x = SAGE_out
        else:
            # Combine SAGE and MLP outputs
            MLP_out = self.MLP(x, *args)
            x = SAGE_out + MLP_out
            
        return x

    def fast_forward(self, x: Tensor, idx: Tensor) -> Tensor:
        """
        Accelerated forward pass using cached GNN outputs.
        
        Args:
            x: Input features
            idx: Node indices to process
            
        Returns:
            Model output for the specified nodes
            
        Raises:
            AssertionError: If gnn_output is not available or MLP is frozen
        """
        # Verify cached output is available
        assert self.gnn_output is not None, "No cached GNN output available"
        assert not self.mlp_freezed, "MLP must be trainable for fast_forward"
        
        # Combine cached GNN output with MLP result
        return (self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + 
                self.MLP(x))