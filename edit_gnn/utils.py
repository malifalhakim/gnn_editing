import torch
import numpy as np
from torch_geometric.data.data import Data
from typing import Dict, List, Tuple, Any, Optional

from models.base import BaseModel

def grab_input(data: Data) -> Dict[str, torch.Tensor]:
    """
    Extract input features and adjacency matrix from data.
    
    Args:
        data: PyG Data object containing graph information
        
    Returns:
        Dictionary with input features and adjacency matrix
    """
    
    return {"x": data.x, 'adj_t': data.adj_t}


@torch.no_grad()
def prediction(model: BaseModel, data: Data) -> torch.Tensor:
    """
    Get model predictions for the given data.
    
    Args:
        model: The GNN model to use for prediction
        data: PyG Data object containing graph information
        
    Returns:
        Model output/logits
    """
    model.eval()
    input_data = grab_input(data)
    return model(**input_data)


def _compute_micro_f1(logits: torch.Tensor, y: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate micro F1 score for model predictions.
    
    Args:
        logits: Model output logits
        y: Ground truth labels
        mask: Optional mask to select specific nodes
        
    Returns:
        Micro F1 score as a float between 0 and 1
    """
    if mask is not None:
        logits, y = logits[mask], y[mask]
    
    # Classification task (single label per node)
    if y.dim() == 1:
        try:
            return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
        except ZeroDivisionError:
            return 0.0
    # Multi-label classification
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.0


@torch.no_grad()
def test(model: BaseModel, data: Data, specific_class: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Test model performance on train/validation/test splits.
    
    Args:
        model: GNN model to evaluate
        data: PyG Data object with graph and masks
        specific_class: Optional class to evaluate performance on
        
    Returns:
        Tuple of (train_accuracy, validation_accuracy, test_accuracy)
    """
    model.eval()
    out = prediction(model, data)
    y_true = data.y
    train_mask = data.train_mask
    valid_mask = data.val_mask
    test_mask = data.test_mask
    
    if specific_class is not None:
        class_mask = data.y == specific_class
        out = out[class_mask]
        y_true = y_true[class_mask]
        train_mask = train_mask[class_mask] if train_mask is not None else None
        valid_mask = valid_mask[class_mask] if valid_mask is not None else None
        test_mask = test_mask[class_mask] if test_mask is not None else None
    
    train_acc = _compute_micro_f1(out, y_true, train_mask)
    valid_acc = _compute_micro_f1(out, y_true, valid_mask)
    test_acc = _compute_micro_f1(out, y_true, test_mask)
    
    return train_acc, valid_acc, test_acc


def select_edit_target_nodes(model: BaseModel,
                           whole_data: Data,
                           num_samples: int,
                           from_valid_set: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select nodes for editing based on incorrect model predictions.
    
    Args:
        model: GNN model to evaluate
        whole_data: PyG Data object with the graph
        num_classes: Number of classes in the classification task
        num_samples: Number of nodes to select for editing
        from_valid_set: Whether to select from validation (True) or training set (False)
        
    Returns:
        Tuple of (node_indices_to_flip, corresponding_labels)
    """
    model.eval()
    bef_edit_logits = prediction(model, whole_data)
    bef_edit_pred = bef_edit_logits.argmax(dim=-1)
    
    # Select from validation or training set
    nodes_set = whole_data.val_mask.nonzero().squeeze() if from_valid_set else whole_data.train_mask.nonzero().squeeze()
    
    # Get validation predictions and ground truth
    val_y_true = whole_data.y[whole_data.val_mask]
    val_y_pred = bef_edit_pred[whole_data.val_mask]
    
    # Find incorrect predictions
    wrong_pred_set = val_y_pred.ne(val_y_true).nonzero()
    
    # Randomly select nodes to flip
    val_node_idx_2flip = wrong_pred_set[torch.randperm(len(wrong_pred_set))[:num_samples]]
    node_idx_2flip = nodes_set[val_node_idx_2flip]
    flipped_label = whole_data.y[node_idx_2flip]

    return node_idx_2flip, flipped_label


def _calculate_drawdowns(before_results: Tuple[float, float, float], 
                        after_accs: List[float]) -> Dict[str, Any]:
    """Helper function to calculate performance drawdowns after editing"""
    bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = before_results
    train_acc, val_acc, test_acc = after_accs[-1], after_accs[-1], after_accs[-1]
    
    tra_drawdown = bef_edit_tra_acc - train_acc
    val_drawdown = bef_edit_val_acc - val_acc
    
    test_drawdowns = np.round(
        (np.array([bef_edit_tst_acc] * len(after_accs)) - np.array(after_accs)),
        decimals=3
    ).tolist()
    
    test_drawdowns_pct = [round(d * 100, 1) for d in test_drawdowns]
    
    return {
        'tra_drawdown': tra_drawdown * 100,
        'val_drawdown': val_drawdown * 100,
        'test_drawdown': test_drawdowns_pct,
        'average_dd': np.round(np.mean(test_drawdowns), decimals=3) * 100,
        'test_dd_std': np.std(test_drawdowns_pct),
        'highest_dd': max(enumerate(test_drawdowns_pct), key=lambda x: x[1]),
        'lowest_dd': min(enumerate(test_drawdowns_pct), key=lambda x: x[1])
    }


def process_edit_results(bef_edit_results: Tuple[float, float, float], 
                       raw_results: List[Tuple]) -> Dict[str, Any]:
    """
    Process and summarize results from graph editing experiments.
    
    Args:
        bef_edit_results: Tuple of (train_acc, val_acc, test_acc) before editing
        raw_results: List of result tuples from multiple edit runs
        
    Returns:
        Dictionary with various performance metrics after editing
    """
    bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
    
    # Unpack results
    train_acc, val_acc, test_acc, succeses, steps, mem, tot_time = zip(*raw_results)
    
    # Calculate drawdowns and statistics
    drawdown_stats = _calculate_drawdowns(bef_edit_results, test_acc)
    
    # Process success rates
    success_list = np.round(np.array(succeses), decimals=3).tolist()
    average_success_rate = np.round(np.mean(succeses), decimals=3).tolist()
    
    # Select specific results for reporting
    selected_result = {
        '1': (drawdown_stats['test_drawdown'][0], success_list[0]),
        '10': (drawdown_stats['test_drawdown'][9], success_list[9]),
        '25': (drawdown_stats['test_drawdown'][24], success_list[24]),
        '50': (drawdown_stats['test_drawdown'][49], success_list[49])
    }
    
    # Memory and time statistics
    mem_result = {
        'max_memory': f"{np.round(np.max(mem), decimals=3)}MB"
    }
    
    time_result = {
        '1': str(np.round(tot_time[0], decimals=3)),
        '10': str(np.round(tot_time[9], decimals=3)),
        '25': str(np.round(tot_time[24], decimals=3)),
        '50': str(np.round(tot_time[49], decimals=3)),
        'total_time': np.sum(tot_time)
    }

    return {
        'bef_edit_tra_acc': bef_edit_tra_acc,
        'bef_edit_val_acc': bef_edit_val_acc,
        'bef_edit_tst_acc': bef_edit_tst_acc,
        **drawdown_stats,
        'average_success_rate': average_success_rate,
        'success_list': success_list,
        'selected_result': selected_result,
        'mean_complexity': np.mean(steps),
        'memory_result': mem_result,
        'time_result': time_result
    }


def success_rate(model: BaseModel, idx: torch.Tensor, label: torch.Tensor, whole_data: Data) -> float:
    """
    Calculate the success rate of model predictions for specific nodes.
    
    Args:
        model: GNN model to evaluate
        idx: Node indices to evaluate
        label: Expected labels for the nodes
        whole_data: PyG Data object with the graph
        
    Returns:
        Success rate as a float between 0 and 1
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    model.eval()

    input_data = grab_input(whole_data)
    
    # Handle different model types
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input_data['x'][idx], idx)
        y_pred = out.argmax(dim=-1)
    else:
        out = model(**input_data)
        y_pred = out.argmax(dim=-1)[idx]
        
    success = int(y_pred.eq(label).sum()) / label.size(0)
    torch.cuda.synchronize()
    
    return success