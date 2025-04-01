import os
import time
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from edit_gnn.utils import grab_input, test
from models.base import BaseModel

logger = logging.getLogger("main")


def get_optimizer(model_config: Dict[str, Any], model: torch.nn.Module, pretrain: bool = False) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model based on configuration.
    
    Args:
        model_config: Configuration dictionary containing optimizer settings
        model: PyTorch model to optimize
        pretrain: Whether to use pretrain learning rate (True) or edit learning rate (False)
        
    Returns:
        Configured PyTorch optimizer
        
    Raises:
        NotImplementedError: If specified optimizer is not supported
    """
    lr = model_config['pretrain_lr'] if pretrain else model_config['edit_lr']
    
    if model_config['optim'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif model_config['optim'] == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer '{model_config['optim']}' not implemented. Use 'adam' or 'rmsprop'.")


def _sorted_checkpoints(
    checkpoint_prefix: str, 
    best_model_checkpoint: Optional[str], 
    output_dir: str
) -> List[str]:
    """
    Get sorted list of model checkpoints, ensuring the best model is at the end.
    
    Args:
        checkpoint_prefix: Prefix of checkpoint files
        best_model_checkpoint: Path to the best model checkpoint
        output_dir: Directory containing checkpoints
        
    Returns:
        Sorted list of checkpoint paths with best model last
    """
    # Find all checkpoint files
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]
    
    # Extract epoch numbers and sort by them
    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    
    # Make sure the best model is at the end (will be kept)
    if best_model_checkpoint is not None and best_model_checkpoint in checkpoints_sorted:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
            checkpoints_sorted[-1],
            checkpoints_sorted[best_model_index],
        )
    
    return checkpoints_sorted


def save_model(model: torch.nn.Module, save_path: str, checkpoint_prefix: str, epoch: int) -> None:
    """
    Save model and manage checkpoint files (keeping only the best one).
    
    Args:
        model: Model to save
        save_path: Directory to save checkpoints
        checkpoint_prefix: Prefix for checkpoint filename
        epoch: Current epoch number for filename
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save current model
    best_model_checkpoint = os.path.join(save_path, f'{checkpoint_prefix}_{epoch}.pt')
    torch.save(model.state_dict(), best_model_checkpoint)
    
    # Get list of checkpoints and delete old ones
    checkpoints_sorted = _sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, save_path)
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    
    # Delete old checkpoints
    for checkpoint in checkpoints_to_be_deleted:
        os.remove(checkpoint)


def _finetune_mlp(
    config: Dict[str, Any], 
    model: BaseModel, 
    whole_data: Data, 
    train_data: Data, 
    batch_size: int, 
    iters: int
) -> None:
    """
    Fine-tune the MLP component of a GNN+MLP model.
    
    Args:
        config: Configuration dictionary
        model: The GNN+MLP model to fine-tune
        whole_data: Complete graph data
        train_data: Training data subset
        batch_size: Batch size for training
        iters: Number of iterations for fine-tuning
    """
    input_data = grab_input(train_data)
    model.eval()

    # Get the original GNN output embedding
    model.mlp_freezed = True
    with torch.no_grad():
        gnn_output = model(**input_data)
        model.gnn_output = model(**grab_input(whole_data)).cpu()
        log_gnn_output = F.log_softmax(gnn_output, dim=-1)

    # Enable MLP training
    model.freeze_module(train=False)
    optimizer = get_optimizer(config['pipeline_params'], model)
    
    # Start fine-tuning
    logger.info('Starting MLP fine-tuning')
    start_time = time.time()
    torch.cuda.synchronize()
    
    device = gnn_output.device
    
    for _ in tqdm(range(iters)):
        optimizer.zero_grad()
        # Sample random batch
        idx = np.random.choice(train_data.num_nodes, batch_size)
        idx = torch.from_numpy(idx).to(device)
        
        # Forward pass
        mlp_output = model.MLP(train_data.x[idx])
        batch_gnn_output = gnn_output[idx]
        
        # Calculate losses
        log_prob = F.log_softmax(mlp_output + batch_gnn_output, dim=-1)
        main_loss = F.cross_entropy(mlp_output + batch_gnn_output, train_data.y[idx])
        kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
        reg_loss = model.MLP.get_regularization_loss()

        # Update weights
        total_loss = kl_loss + main_loss + reg_loss
        total_loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.time()
    logger.info(f'MLP fine-tuning completed in {end_time - start_time:.2f} seconds')


def finetune_gnn_mlp(
    config: Dict[str, Any], 
    model: BaseModel, 
    whole_data: Data, 
    train_data: Data
) -> Tuple[float, float, float]:
    """
    Fine-tune a GNN+MLP model with dataset-specific settings.
    
    Args:
        config: Configuration dictionary
        model: The model to fine-tune
        whole_data: Complete graph data
        train_data: Training data subset
        
    Returns:
        Tuple of (train_accuracy, validation_accuracy, test_accuracy) after fine-tuning
    """
    model.freeze_module(train=False)
    dataset = config['eval_params']['dataset']
    
    # Use larger batch size for specific datasets
    use_large_batch = dataset == 'flickr' or \
                     (dataset == 'reddit2' and config['pipeline_params']['model_name']) or \
                     (dataset in ['amazoncomputers', 'amazonphoto', 'coauthorcs', 'coauthorphysics'])
    
    batch_size = 512 if use_large_batch else 32
    
    # Perform fine-tuning
    _finetune_mlp(
        config=config, 
        model=model, 
        whole_data=whole_data, 
        train_data=train_data, 
        batch_size=batch_size, 
        iters=100
    )
    
    # Evaluate model after fine-tuning
    fine_tuned_results = test(model, whole_data)
    ft_train_acc, ft_valid_acc, ft_test_acc = fine_tuned_results
    
    logger.info(
        f'After fine-tuning: Train acc: {ft_train_acc:.4f}, '
        f'Valid acc: {ft_valid_acc:.4f}, Test acc: {ft_test_acc:.4f}'
    )

    return fine_tuned_results


def _check_prediction_before_edit(
    model: BaseModel, 
    whole_data: Data, 
    idx: torch.Tensor, 
    label: torch.Tensor, 
    curr_edit_target: int
) -> float:
    """
    Check if the model already correctly predicts the target label before editing.
    
    Args:
        model: Model to evaluate
        whole_data: Complete graph data
        idx: Node indices to check
        label: Target labels
        curr_edit_target: Index of current edit target within the batch
        
    Returns:
        Success rate (1.0 if successful, 0.0 otherwise)
    """
    model.eval()
    torch.cuda.synchronize()
    input_data = grab_input(whole_data)
    
    # Handle different model types
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
        out = model.fast_forward(input_data['x'][idx], idx)
        y_pred = out.argmax(dim=-1)
    else:
        out = model(**input_data)
        y_pred = out.argmax(dim=-1)[idx]

    # Calculate success
    if label.shape[0] == 1:
        success = float(y_pred == label)
    else:
        success = 1.0 if y_pred.eq(label)[curr_edit_target] else 0.0
    
    torch.cuda.synchronize()
    return success


def _single_edit(
    model: BaseModel, 
    whole_data: Data, 
    idx: torch.Tensor, 
    label: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    max_num_step: int, 
    num_edit_targets: int = 1
) -> Tuple[BaseModel, float, torch.Tensor, int, float, float]:
    """
    Perform a single editing operation on the model.
    
    Args:
        model: Model to edit
        whole_data: Complete graph data
        idx: Node indices to edit
        label: Target labels
        optimizer: Optimizer for model parameters
        max_num_step: Maximum number of optimization steps
        num_edit_targets: Number of target nodes to edit
        
    Returns:
        Tuple of (edited_model, success_rate, loss, steps_taken, memory_used, time_taken)
    """
    start_time = time.time()
    loss_op = F.cross_entropy
    
    # Reset CUDA memory tracking
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    success = 0.0
    step = 0
    loss = None

    # Optimization loop
    for step in range(1, max_num_step + 1):
        optimizer.zero_grad()
        input_data = grab_input(whole_data)
        
        # Forward pass (different for GNN+MLP models)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP']:
            out = model.fast_forward(input_data['x'][idx], idx)
            loss = loss_op(out, label)
            y_pred = out.argmax(dim=-1)
        else:
            out = model(**input_data)
            loss = loss_op(out[idx], label)
            y_pred = out.argmax(dim=-1)[idx]
            
        # Update weights
        loss.backward()
        optimizer.step()
        
        # Check for success
        if label.shape[0] == 1:
            success = float(y_pred == label)
        else:
            success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets]).sum()) / num_edit_targets
            
        # Stop if all targets are correctly predicted
        if success == 1.0:
            break
    
    torch.cuda.synchronize()
    end_time = time.time()
    memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    time_taken = end_time - start_time
    
    logger.info(f'Edit memory usage: {memory_used:.2f} MB')
    logger.info(f'Edit time: {time_taken:.4f} seconds')
    
    return model, success, loss, step, memory_used, time_taken


def edit(
    model: BaseModel, 
    whole_data: Data, 
    idx: torch.Tensor, 
    f_label: torch.Tensor, 
    optimizer: torch.optim.Optimizer, 
    max_num_step: int,
    num_edit_targets: int = 1, 
    curr_edit_target: int = 0
) -> Tuple[BaseModel, float, torch.Tensor, int, float, float]:
    """
    Edit a model to correctly predict labels for target nodes.
    First checks if model already predicts correctly; if so, skips editing.
    
    Args:
        model: Model to edit
        whole_data: Complete graph data
        idx: Node indices to edit
        f_label: Target labels
        optimizer: Optimizer for model parameters
        max_num_step: Maximum number of optimization steps
        num_edit_targets: Number of target nodes to edit
        curr_edit_target: Index of current edit target within the batch
        
    Returns:
        Tuple of (edited_model, success_rate, loss, steps_taken, memory_used, time_taken)
    """
    # Check if model already predicts correctly
    pre_edit_success = _check_prediction_before_edit(model, whole_data, idx, f_label, curr_edit_target)
    
    if pre_edit_success == 1.0:
        return model, pre_edit_success, 0, 0, 0, 0

    # Perform editing if needed
    return _single_edit(
        model=model, 
        whole_data=whole_data, 
        idx=idx, 
        label=f_label, 
        optimizer=optimizer, 
        max_num_step=max_num_step, 
        num_edit_targets=num_edit_targets
    )


def process_raw_exp_results(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw experimental results into a more concise format.
    
    Args:
        raw_results: Dictionary containing raw experimental results
        
    Returns:
        Dictionary with selected key results
    """
    return {
        'bef_edit_tst_acc': raw_results['bef_edit_tst_acc'],
        'selected_result': raw_results['selected_result'],
        'highest_dd': raw_results['highest_dd'],
        'average_dd': raw_results['average_dd'],
        'average_success_rate': raw_results['average_success_rate']
    }