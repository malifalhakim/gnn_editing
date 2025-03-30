import os
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data

import models
from constants import SEED
from main_utils import set_seeds_all
from models.base import BaseModel
from data import get_data, prepare_dataset
from edit_gnn.utils import grab_input, test
from pipelines.seed_gnn.utils import get_optimizer, save_model
from pipelines.seed_gnn.seed_gnn_logging import Logger

logger = logging.getLogger("main")

def _train_loop(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_data: Data,
    loss_op: Callable
) -> float:
    """
    Execute one training iteration with the model.
    
    Args:
        model: The GNN model being trained
        optimizer: Optimizer for model parameters
        train_data: Training data containing graph structure and features
        loss_op: Loss function to optimize
        
    Returns:
        float: Training loss value for this iteration
    """
    model.train()
    optimizer.zero_grad()
    model_input = grab_input(train_data)
    output = model(**model_input)
    loss = loss_op(output[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def _initialize_model(
    config: Dict[str, Any], 
    num_features: int, 
    num_classes: int
) -> BaseModel:
    """
    Initialize model based on configuration.
    
    Args:
        config: Configuration dictionary
        num_features: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Initialized model
    """
    model_name = config['pipeline_params']['model_name']
    MODEL_FAMILY = getattr(models, model_name)
    save_path = os.path.join(config['management']['pretrain_output_dir'], 
                            config['eval_params']['dataset'])
    
    model = MODEL_FAMILY(
        in_channels=num_features, 
        out_channels=num_classes, 
        load_pretrained_backbone=config['pipeline_params']['load_pretrained_backbone'], 
        saved_ckpt_path=save_path, 
        **config['pipeline_params']['architecture']
    )
    model.cuda()
    
    # Reset parameters if not loading pretrained model
    if not config['pipeline_params']['load_pretrained_backbone']:
        model.reset_parameters()
        
    return model


def _train_model(config: Dict[str, Any]) -> None:
    """
    Train a GNN model according to configuration.
    
    Args:
        config: Configuration dictionary with training parameters
    """
    # Set seeds for reproducibility
    set_seeds_all(SEED)
    
    # Load dataset
    data, num_features, num_classes = get_data(
        config['management']['dataset_dir'], 
        config['eval_params']['dataset']
    )
    
    # Initialize model
    model = _initialize_model(config, num_features, num_classes)
    logger.info(f"Model architecture:\n{model}")
    
    # Prepare dataset
    train_data, whole_data = prepare_dataset(
        config['pipeline_params'], 
        data, 
        remove_edge_index=True
    )
    del data
    logger.info(f'Training data: {train_data}')
    logger.info(f'Whole data: {whole_data}')
    
    # Set up training
    optimizer = get_optimizer(config['pipeline_params'], model, pretrain=True)
    loss_op = F.cross_entropy
    train_logger = Logger()
    
    # Paths for saving model
    checkpoint_prefix = f"{config['pipeline_params']['model_name']}"
    save_path = os.path.join(
        config['management']['pretrain_output_dir'], 
        config['eval_params']['dataset']
    )
    
    # Track best validation accuracy for model saving
    best_val_acc = -1.0
    
    # Training loop
    for epoch in range(1, config['pipeline_params']['epochs'] + 1):
        # Skip training if using pretrained backbone
        if not config['pipeline_params']['load_pretrained_backbone']:
            train_loss = _train_loop(model, optimizer, train_data, loss_op)
        
        # Evaluate model
        result = test(model, whole_data)
        train_logger.add_result(result)
        train_acc, valid_acc, test_acc = result
        
        # Save model if validation accuracy improves
        if valid_acc > best_val_acc:
            save_model(model, save_path, checkpoint_prefix, epoch)
            best_val_acc = valid_acc
        
        # Log progress
        logger.info(
            f'Epoch: {epoch:02d}, '
            f'Train F1: {100 * train_acc:.2f}%, '
            f'Valid F1: {100 * valid_acc:.2f}%, '
            f'Test F1: {100 * test_acc:.2f}%'
        )
    
    # Print final statistics
    train_logger.print_statistics()


def pretrain_gnn(config: Dict[str, Any]) -> None:
    """
    Pretrains a GNN or GNN+MLP model, ensuring the backbone is trained first if needed.
    
    Args:
        config: Configuration dictionary with model and training parameters
    """
    model_name = config['pipeline_params']['model_name']
    save_path = os.path.join(
        config['management']['pretrain_output_dir'], 
        config['eval_params']['dataset']
    )
    
    # Check for existing checkpoints
    backbone_name = model_name.replace('_MLP', '')
    checkpoints = list(Path(save_path).glob(f"{backbone_name}_*.pt"))
    
    # For GNN + MLP models, pretrain the GNN backbone first if needed
    if '_MLP' in model_name and len(checkpoints) < 1:
        logger.info(f"No pretrained backbone found for {model_name}. Training backbone first.")
        backbone_config = deepcopy(config)
        backbone_config['pipeline_params']['model_name'] = backbone_name
        backbone_config['pipeline_params']['load_pretrained_backbone'] = False
        _train_model(backbone_config)
    
    # Train the full model (or just the main model if not GNN+MLP)
    logger.info(f"Training {'full model' if '_MLP' in model_name else 'model'}: {model_name}")
    _train_model(config)