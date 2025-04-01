from torch import Tensor
from torch.nn import Linear, ModuleList, LayerNorm, Parameter
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

from .base import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron implementation with enhanced regularization.
    
    A standard MLP with configurable depth, hidden dimensions,
    and multiple regularization options including dropout, batch normalization,
    residual connections, layer normalization, spectral normalization,
    and activity regularization.
    
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
                 saved_ckpt_path: str = '',
                 layer_norm: bool = False,
                 spectral_norm_enabled: bool = False,
                 weight_decay: float = 0.0,
                 activity_regularization: float = 0.0,
                 dropout_type: str = 'standard'):  # 'standard', 'alpha', or 'concrete'
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
            layer_norm: Whether to use layer normalization
            spectral_norm_enabled: Whether to use spectral normalization
            weight_decay: L2 regularization strength (used during optimization)
            activity_regularization: L1/L2 penalty on activations
            dropout_type: Type of dropout to use ('standard', 'alpha', or 'concrete')
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
        
        # Store additional regularization parameters
        self.layer_norm = layer_norm
        self.spectral_norm_enabled = spectral_norm_enabled
        self.activity_regularization = activity_regularization
        self.dropout_type = dropout_type
        self.weight_decay = weight_decay
        
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
            
            # Apply spectral normalization if enabled
            if spectral_norm_enabled and i < num_layers - 1:  # Don't apply to last layer
                lin = spectral_norm(lin)
                
            self.lins.append(lin)
        
        # Initialize layer normalization layers if enabled
        if self.layer_norm:
            self.lns = ModuleList()
            for i in range(num_layers - 1):  # One less than linear layers
                self.lns.append(LayerNorm(hidden_channels))
        
        # Alpha dropout parameters if using alpha dropout
        if dropout_type == 'alpha':
            self.alpha = Parameter(torch.ones(1))
        
        # Store activity regularization loss
        self.activity_reg_loss = 0.0

    def reset_parameters(self) -> None:
        """Reset all model parameters with improved initialization."""
        for lin in self.lins:
            # Use Kaiming initialization for better gradient flow
            if hasattr(lin, 'weight'):
                torch.nn.init.kaiming_normal_(lin.weight, mode='fan_out', nonlinearity='relu')
                if lin.bias is not None:
                    torch.nn.init.zeros_(lin.bias)
            else:
                lin.reset_parameters()
            
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        
        if self.layer_norm:
            for ln in self.lns:
                ln.reset_parameters()
                
        # Reset activity regularization loss
        self.activity_reg_loss = 0.0

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass through the MLP with enhanced regularization.
        
        Args:
            x: Input feature tensor
            *args: Additional positional arguments 
            **kwargs: Additional keyword arguments 
            
        Returns:
            Output tensor with shape [batch_size, out_channels]
        """
        # Reset activity regularization loss at the beginning of forward pass
        self.activity_reg_loss = 0.0
        
        # Process through all layers except the last one
        for idx in range(self.num_layers - 1):
            lin = self.lins[idx]
            h = lin(x, *args, **kwargs) 
            
            # Apply batch normalization if specified
            if self.batch_norm:
                h = self.bns[idx](h)
            
            # Apply layer normalization if specified
            if self.layer_norm:
                h = self.lns[idx](h)
                
            # Apply residual connection if dimensions match
            if self.residual and h.size(-1) == x.size(-1):
                # Use minimum size to avoid dimension mismatch
                min_size = min(h.size(0), x.size(0))
                h[:min_size] += x[:min_size]
            
            # Add activity regularization (L2 penalty on activations)
            if self.activity_regularization > 0:
                self.activity_reg_loss += self.activity_regularization * torch.sum(h**2)
                
            # Apply activation
            x = self.activation(h)
            
            # Apply different types of dropout
            if self.dropout_type == 'standard':
                x = self.dropout(x)
            elif self.dropout_type == 'alpha':
                if self.training:
                    x = x * (self.alpha * torch.randn_like(x) + 1.0)
            elif self.dropout_type == 'concrete':
                if self.training and self.dropout.p > 0:
                    # Concrete dropout implementation
                    eps = 1e-7
                    temp = 0.1
                    unif_noise = torch.rand_like(x)
                    drop_prob = self.dropout.p
                    
                    drop_mask = torch.log(drop_prob + eps) - torch.log(1 - drop_prob + eps) + \
                               torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)
                    drop_mask = torch.sigmoid(drop_mask / temp)
                    x = x * (1 - drop_mask) / (1 - drop_prob)
            
        # Process the last layer (without activation)
        x = self.lins[-1](x, *args, **kwargs)
        return x
    
    def get_regularization_loss(self) -> Tensor:
        """
        Get the total regularization loss for use in training.
        
        Returns:
            Tensor containing the regularization loss
        """
        reg_loss = self.activity_reg_loss
        
        # Add L2 weight decay if specified
        if self.weight_decay > 0:
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in self.parameters():
                l2_reg = l2_reg + torch.norm(param)**2
            reg_loss += self.weight_decay * l2_reg
            
        return reg_loss

    def freeze_layer(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze all parameters of the MLP.
        
        Args:
            freeze: If True, freeze parameters. If False, unfreeze parameters.
        """
        for param in self.parameters():
            param.requires_grad = not freeze