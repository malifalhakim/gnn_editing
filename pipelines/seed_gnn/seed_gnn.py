from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Any, Optional, Union

import torch
from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph

from main_utils import set_seeds_all
from models.base import BaseModel
from edit_gnn.utils import test, success_rate, prediction
from pipelines.seed_gnn.utils import get_optimizer, edit


def _select_mixup_training_nodes(
    model: BaseModel,
    whole_data: Data,
    alpha: float,
    num_samples: int = 0,
    center_node_idx: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select nodes for mixup training based on model predictions and neighborhood.
    
    Args:
        model: The GNN model to use
        whole_data: Complete graph data
        alpha: Ratio of neighborhood nodes to include (0-1)
        num_samples: Total number of samples to select
        center_node_idx: Center node to extract neighborhood from
        
    Returns:
        Tuple of (selected_node_indices, corresponding_labels)
    """
    # Get current model predictions
    bef_edit_logits = prediction(model, whole_data)
    bef_edit_pred = bef_edit_logits.argmax(dim=-1)
    
    # Get training set information
    train_y_true = whole_data.y[whole_data.train_mask]
    train_y_pred = bef_edit_pred[whole_data.train_mask]
    nodes_set = whole_data.train_mask.nonzero().squeeze()
    
    # Find correctly predicted nodes
    right_pred_set = train_y_pred.eq(train_y_true).nonzero()
    device = right_pred_set.device

    # Extract neighborhood of center node with increasing hops until we have enough neighbors
    neighbors = torch.Tensor([])
    num_hop = 0
    while len(neighbors) < num_samples and num_hop < 4:
        num_hop += 1
        neighbors, _, _, _ = k_hop_subgraph(center_node_idx, num_hops=num_hop, 
                                          edge_index=whole_data.edge_index)
    
    # Select correct predictions from the neighborhood
    correct_neighbors_pred_set = train_y_pred.eq(train_y_true).nonzero().to(device)
    correct_neighbors_pred_set = correct_neighbors_pred_set.squeeze().cpu().numpy().tolist()
    
    # Convert to tensor and filter to only include nodes that are neighbors
    correct_neighbors = torch.tensor(
        [int(i) for i in correct_neighbors_pred_set if i in neighbors],
        dtype=torch.long, device=device
    ).unsqueeze(dim=1)
    
    # Select a mix of neighborhood nodes and general correctly predicted nodes
    num_neighborhood_nodes = int(num_samples * alpha)
    num_general_nodes = num_samples - num_neighborhood_nodes
    
    # Random sample from neighborhood nodes
    neighborhood_selection = correct_neighbors[
        torch.randperm(len(correct_neighbors))[:num_neighborhood_nodes]
    ].to(device)
    
    # Random sample from general correctly predicted nodes
    general_selection = right_pred_set[
        torch.randperm(len(right_pred_set))[:num_general_nodes]
    ]
    
    # Combine both selections
    train_mixup_indices = torch.cat((neighborhood_selection, general_selection), dim=0)
    
    # Map back to original node indices and get labels
    mixup_training_samples_idx = nodes_set[train_mixup_indices]
    mixup_label = whole_data.y[mixup_training_samples_idx]

    return mixup_training_samples_idx, mixup_label


def seed_gnn_edit(
    config: Dict[str, Any],
    model: BaseModel,
    node_idx_2flip: torch.Tensor,
    flipped_label: torch.Tensor,
    whole_data: Data,
    max_num_step: int
) -> List[List[Any]]:
    """
    Perform graph neural network editing using the SEED-GNN approach.
    
    Args:
        config: Configuration dictionary with parameters
        model: GNN model to edit
        node_idx_2flip: Node indices to be edited
        flipped_label: Target labels for edit nodes
        whole_data: Complete graph data
        max_num_step: Maximum number of optimization steps per edit
        
    Returns:
        List of results for each edit containing:
        [train_acc, val_acc, test_acc, success_rate, steps, memory_usage, total_time]
    """
    model.train()
    # Keep original model for node selection, edit a copy
    original_model = model
    model = deepcopy(model)
    optimizer = get_optimizer(config['pipeline_params'], model)
    raw_results = []

    # Process each target node sequentially
    for idx in tqdm(range(len(node_idx_2flip))):
        # Set seed based on index for reproducibility
        set_seeds_all(idx)
        
        # Select additional training nodes using mixup approach
        mixup_training_samples_idx, mixup_label = _select_mixup_training_nodes(
            model=original_model, 
            whole_data=whole_data,
            alpha=config['pipeline_params']['alpha'],
            num_samples=config['pipeline_params']['beta'],
            center_node_idx=node_idx_2flip[idx]
        )
        
        # Combine edit targets processed so far with mixup nodes
        all_nodes = torch.cat((
            node_idx_2flip[:idx+1].squeeze(dim=1), 
            mixup_training_samples_idx.squeeze(dim=1)
        ), dim=0)
        
        all_labels = torch.cat((
            flipped_label[:idx+1].squeeze(dim=1), 
            mixup_label.squeeze(dim=1)
        ), dim=0)

        # Perform model editing
        edited_model, edit_success, loss, steps, mem, tot_time = edit(
            model=model,
            whole_data=whole_data,
            idx=all_nodes,
            f_label=all_labels,
            optimizer=optimizer,
            max_num_step=max_num_step,
            num_edit_targets=idx + 1,
            curr_edit_target=idx
        )

        # Calculate success rate on all edited nodes so far
        current_success = success_rate(
            model=model, 
            idx=node_idx_2flip[:idx+1].squeeze(dim=1), 
            label=flipped_label[:idx+1].squeeze(dim=1), 
            whole_data=whole_data
        )
        
        # Test model performance
        train_acc, val_acc, test_acc = test(edited_model, whole_data)
        
        # Compile results
        result = [
            train_acc,    # Train accuracy
            val_acc,      # Validation accuracy
            test_acc,     # Test accuracy
            current_success,  # Success rate
            steps,        # Number of steps
            mem,          # Memory usage
            tot_time      # Total editing time
        ]
        raw_results.append(result)
    
    return raw_results