from torch import Tensor
from torch.nn import Linear, ModuleList

from .base import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron implementation.
    
    A standard MLP with configurable depth, hidden dimensions,
    and optional residual connections and batch normalization.
    
    Attributes:
        lins (ModuleList): List of linear layers
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        """
        Initialize the MLP model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of linear layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            load_pretrained_backbone: Whether to load pretrained weights
            saved_ckpt_path: Path to saved checkpoint
        """
        super(MLP, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual
        )
        
        # Initialize linear layers
        self.lins = ModuleList()
        for i in range(num_layers):
            # Calculate input and output dimensions based on layer position
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                
            # Create linear layer
            lin = Linear(in_features=in_dim, out_features=out_dim, bias=True)
            self.lins.append(lin)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        for lin in self.lins:
            lin.reset_parameters()
            
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input feature tensor
            *args: Additional positional arguments 
            **kwargs: Additional keyword arguments 
            
        Returns:
            Output tensor with shape [batch_size, out_channels]
        """
        # Process through all layers except the last one
        for idx in range(self.num_layers - 1):
            lin = self.lins[idx]
            h = lin(x, *args, **kwargs) 
            
            # Apply batch normalization if specified
            if self.batch_norm:
                h = self.bns[idx](h)
                
            # Apply residual connection if dimensions match
            if self.residual and h.size(-1) == x.size(-1):
                # Use minimum size to avoid dimension mismatch
                min_size = min(h.size(0), x.size(0))
                h[:min_size] += x[:min_size]
                
            # Apply activation and dropout
            x = self.activation(h)
            x = self.dropout(x)
            
        # Process the last layer (without activation)
        x = self.lins[-1](x, *args, **kwargs)
        return x

    def freeze_layer(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze all parameters of the MLP.
        
        Args:
            freeze: If True, freeze parameters. If False, unfreeze parameters.
        """
        for param in self.parameters():
            param.requires_grad = not freeze