import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

from tqdm import tqdm


class BaseModel(torch.nn.Module):
    """
    Base class for all GNN models in the framework.
    
    This class provides common functionality including initialization,
    loading from checkpoints, and parameter reset.
    
    Attributes:
        in_channels (int): Number of input features
        hidden_channels (int): Number of hidden features
        out_channels (int): Number of output features/classes
        num_layers (int): Number of layers in the model
        dropout (torch.nn.Dropout): Dropout layer
        activation (torch.nn.Module): Activation function
        batch_norm (bool): Whether to use batch normalization
        bns (ModuleList): List of BatchNorm layers if batch_norm is True
        residual (bool): Whether to use residual connections
        use_linear (bool): Whether to use linear layers alongside GNN layers
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
        Initialize the base model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of layers in the model
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            use_linear: Whether to use linear layers alongside GNN layers
        """
        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_layers = num_layers
        self.use_linear = use_linear
        
        # Initialize batch normalization layers if needed
        if self.batch_norm:
            self.bns = ModuleList()
            for _ in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)

    @classmethod
    def from_pretrained(cls, 
                        in_channels: int, 
                        out_channels: int, 
                        saved_ckpt_path: str, 
                        **kwargs) -> 'BaseModel':
        """
        Create a model instance and load weights from a checkpoint.
        
        Args:
            in_channels: Number of input features
            out_channels: Number of output features/classes
            saved_ckpt_path: Path to the checkpoint file or directory
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Model with loaded weights
            
        Raises:
            FileNotFoundError: If no checkpoint file is found
            AssertionError: If multiple checkpoint files are found
        """
        # Create model instance
        model = cls(in_channels=in_channels, out_channels=out_channels, **kwargs)
        
        # Handle checkpoint path (file or directory)
        if not saved_ckpt_path.endswith('.pt'):
            # Handle Lora models specially - they use the base model's weights
            class_name = cls.__name__
            if '_Lora' in class_name:
                base_class_name = class_name.replace('_Lora', '')
                checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{base_class_name}_*.pt")]
            else:
                checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{class_name}_*.pt")]
            
            # Filter checkpoints based on MLP suffix
            if '_MLP' not in class_name:
                glob_checkpoints = [x for x in checkpoints if '_MLP' not in x]
            else:
                glob_checkpoints = checkpoints
            
            # Ensure exactly one checkpoint is found
            if not glob_checkpoints:
                raise FileNotFoundError(f"No checkpoint found for {class_name} in {saved_ckpt_path}")
            if len(glob_checkpoints) > 1:
                raise AssertionError(
                    f"Multiple checkpoints found for {class_name} in {saved_ckpt_path}: {glob_checkpoints}"
                )
            
            saved_ckpt_path = glob_checkpoints[0]
        
        print(f'Loading model weights from {saved_ckpt_path}')
        state_dict = torch.load(saved_ckpt_path, map_location='cpu')
        
        # Process state dict - handle model prefixes and ignore specific keys
        final_state_dict = {}
        ignore_keys = ['edit_lrs']
        for k, v in state_dict.items():
            if k in ignore_keys:
                continue
            if k.startswith('model.'):
                new_k = k.split('model.')[1]
                final_state_dict[new_k] = v
            else:
                final_state_dict[k] = v
        
        # Load weights into model
        model.load_state_dict(final_state_dict, strict=False)
        return model

    def reset_parameters(self) -> None:
        """Reset model parameters. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement reset_parameters")


class BaseGNNModel(BaseModel):
    """
    Base class for Graph Neural Network models.
    
    Extends BaseModel with graph-specific operations including
    convolution layers and specialized forward passes.
    
    Attributes:
        convs (ModuleList): List of graph convolution layers
        lins (torch.nn.ModuleList): Optional list of linear layers
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
        Initialize the GNN model.
        
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features/classes
            num_layers: Number of layers in the model
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            use_linear: Whether to use linear layers alongside GNN layers
        """
        super(BaseGNNModel, self).__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels, 
            num_layers=num_layers,
            dropout=dropout, 
            batch_norm=batch_norm, 
            residual=residual, 
            use_linear=use_linear
        )
        
        # Initialize layers
        self.convs = ModuleList()
        if self.use_linear:
            self.lins = torch.nn.ModuleList()

    def reset_parameters(self) -> None:
        """Reset all model parameters (convolutions and batch norms)."""
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
                
        if self.use_linear:
            for lin in self.lins:
                lin.reset_parameters()

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        """
        Forward pass for the GNN model.
        
        Args:
            x: Node feature matrix
            adj_t: Sparse adjacency tensor
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor with shape [num_nodes, out_channels]
        """
        # Process all layers except the last one
        for idx in range(self.num_layers - 1):
            conv = self.convs[idx]
            h = conv(x, adj_t)
            
            # Apply linear transformation if specified
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
                
            # Apply batch normalization if specified
            if self.batch_norm:
                h = self.bns[idx](h)
                
            # Add residual connection if dimensions match
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
                
            # Apply activation and dropout
            x = self.activation(h)
            x = self.dropout(x)
        
        # Process the last layer
        h = self.convs[-1](x, adj_t, *args, **kwargs)
        
        # Apply final linear transformation if specified
        if self.use_linear:
            linear = self.lins[-1](x)
            x = h + linear
        else:
            x = h
            
        return x

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor, 
                      size: Tuple[int, int]) -> Tensor:
        """
        Forward pass for a single layer of the GNN.
        
        Args:
            layer: Layer index
            x: Node feature matrix
            adj_t: Sparse adjacency tensor
            size: Tuple of (source_size, target_size)
            
        Returns:
            Output tensor for the specified layer
            
        Raises:
            NotImplementedError: If use_linear is True
        """
        if self.use_linear:
            raise NotImplementedError("forward_layer not implemented for models with linear layers")
        
        # Apply dropout for all but the first layer
        if layer != 0:
            x = self.dropout(x)
            
        # Get target nodes
        x_target = x[:size[1]]
        
        # Apply convolution
        h = self.convs[layer]((x, x_target), adj_t)
        
        # Apply batch normalization, residual connection and activation
        # for all but the last layer
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = F.relu(h)
            
        return h

    @torch.no_grad()
    def mini_inference(self, x_all: Tensor, loader) -> Tensor:
        """
        Memory-efficient inference for large graphs.
        
        Args:
            x_all: Full node feature matrix
            loader: Batch loader that yields (batch_size, node_ids, adjacency)
            
        Returns:
            Model output for all nodes
        """
        # Set up progress bar
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        
        # Process each layer
        for i in range(len(self.convs)):
            xs = []
            # Process each batch
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index, size).cpu())
                pbar.update(batch_size)
                
            # Concatenate results for this layer
            x_all = torch.cat(xs, dim=0)
            
        pbar.close()
        return x_all