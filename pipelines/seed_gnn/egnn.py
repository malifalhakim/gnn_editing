from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Any
import torch
from torch_geometric.data import Data

from edit_gnn.utils import test, success_rate
from pipelines.seed_gnn.utils import get_optimizer, edit

def egnn_edit(
    config: Dict[str, Any],
    model: torch.nn.Module,
    node_idx_2flip: torch.Tensor,
    flipped_label: torch.Tensor,
    whole_data: Data,
    max_num_step: int
) -> List[List[Any]]:
    """
    Edit a GNN model by iteratively modifying its weights based on target node labels.
    
    Args:
        config: Configuration dictionary containing pipeline parameters
        model: The GNN model to be edited
        node_idx_2flip: Indices of nodes to be edited
        flipped_label: Target labels for the nodes to edit
        whole_data: The complete graph data
        max_num_step: Maximum number of steps for each edit operation
        
    Returns:
        List of results for each edit step, containing:
        [train_acc, val_acc, test_acc, success_rate, steps, memory_usage, total_time]
    """
    # Start with a deep copy of the model to preserve the original
    model_copy = deepcopy(model)
    model_copy.train()
    
    # Initialize optimizer
    optimizer = get_optimizer(config['pipeline_params'], model_copy)
    raw_results = []

    # Process each node to edit with progress bar
    for i, (idx, f_label) in enumerate(tqdm(zip(node_idx_2flip, flipped_label), total=len(node_idx_2flip))):
        # Edit the model for this specific node
        edited_model, edit_success, loss, steps, mem, tot_time = edit(
            model_copy, whole_data, idx, f_label, optimizer, max_num_step
        )
        
        # Calculate cumulative success rate on all edited nodes so far
        nodes_edited_so_far = node_idx_2flip[:i+1].squeeze(dim=1) 
        labels_so_far = flipped_label[:i+1].squeeze(dim=1)
        success = success_rate(model_copy, nodes_edited_so_far, labels_so_far, whole_data)
        
        # Test model performance and collect results
        train_acc, val_acc, test_acc = test(edited_model, whole_data)
        
        # Combine all metrics
        result = [
            train_acc,   # Training accuracy
            val_acc,     # Validation accuracy
            test_acc,    # Test accuracy 
            success,     # Success rate on edited nodes
            steps,       # Number of steps taken
            mem,         # Memory usage
            tot_time     # Total time taken
        ]
        raw_results.append(result)
    
    return raw_results
