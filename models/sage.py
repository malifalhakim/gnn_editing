import torch
from torch_geometric.nn import SAGEConv
from .base import BaseGNNModel


class SAGE(BaseGNNModel):
    """
    Graph SAGE (SAmple and aggreGatE) implementation.
    
    Implementation of GraphSAGE as described in the paper
    "Inductive Representation Learning on Large Graphs" by Hamilton et al.
    
    This model performs neighborhood aggregation to generate node embeddings.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int,
                 out_channels: int, 
                 num_layers: int, 
                 dropout: float = 0.0,
                 batch_norm: bool = False, 
                 residual: bool = False, 
                 use_linear: bool = False):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            use_linear: Whether to use additional linear layers
            load_pretrained_backbone: Whether to load pretrained weights
            saved_ckpt_path: Path to the pretrained weights
        """
        super(SAGE, self).__init__(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            num_layers=num_layers, 
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual, 
            use_linear=use_linear
        )
        
        # Initialize SAGE convolution layers
        for i in range(num_layers):
            # Calculate input and output dimensions based on layer position
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                
            # Create SAGE convolution layer
            conv = SAGEConv(in_dim, out_dim)
            self.convs.append(conv)
            
            # Add optional linear projection
            if self.use_linear:
                self.lins.append(torch.nn.Linear(in_dim, out_dim, bias=False))