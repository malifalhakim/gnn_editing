import os
import logging
from typing import Dict, Any, Tuple, Optional, List, Union

import models as models
from main_utils import set_seeds_all
from data import get_data, prepare_dataset
from constants import SEED
from edit_gnn.utils import test, select_edit_target_nodes, process_edit_results
from pipelines.seed_gnn.utils import finetune_gnn_mlp, process_raw_exp_results
from pipelines.seed_gnn.egnn import egnn_edit
from pipelines.seed_gnn.seed_gnn import seed_gnn_edit

logger = logging.getLogger("main")

def _load_model(config: Dict[str, Any], num_features: int, num_classes: int):
    """
    Load and initialize the model based on configuration.
    
    Args:
        config: Dictionary containing configuration parameters
        num_features: Number of input features
        num_classes: Number of output classes
        
    Returns:
        Initialized model
    """
    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    save_path = os.path.join(
        config['management']['pretrain_output_dir'], 
        config['eval_params']['dataset']
    )
    
    model = MODEL_FAMILY.from_pretrained(
        in_channels=num_features,
        out_channels=num_classes,
        saved_ckpt_path=save_path,
        **config['pipeline_params']['architecture']
    )
    
    logger.info(model)
    model.cuda()
    return model

def _run_edit_method(
    config: Dict[str, Any], 
    model: Any, 
    node_idx_2flip: Any, 
    flipped_label: Any, 
    whole_data: Any
) -> List[List[Any]]:
    """
    Run the specified graph editing method.
    
    Args:
        config: Configuration dictionary
        model: Model to be edited
        node_idx_2flip: Target node indices to edit
        flipped_label: Target labels for the edited nodes
        whole_data: Complete graph data
        
    Returns:
        Raw results from the editing method
        
    Raises:
        ValueError: If the specified editing method is not implemented
    """
    method = config['pipeline_params']['method']
    max_steps = config['pipeline_params']['max_num_edit_steps']
    
    if method == 'egnn':
        return egnn_edit(
            config=config,
            model=model,
            node_idx_2flip=node_idx_2flip,
            flipped_label=flipped_label,
            whole_data=whole_data,
            max_num_step=max_steps
        )
    elif method == 'seed_gnn':
        return seed_gnn_edit(
            config=config,
            model=model,
            node_idx_2flip=node_idx_2flip,
            flipped_label=flipped_label,
            whole_data=whole_data,
            max_num_step=max_steps
        )
    else:
        error_msg = f"Editing method '{method}' is not implemented. Available methods: 'egnn', 'seed_gnn'"
        logger.error(error_msg)
        raise ValueError(error_msg)

def eval_edit_gnn(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate graph neural network editing methods on a dataset.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Tuple of (raw_results, processed_results) from the editing experiment
    """
    # Set random seeds for reproducibility
    set_seeds_all(SEED)
    
    # Load dataset
    data, num_features, num_classes = get_data(
        config['management']['dataset_dir'], 
        config['eval_params']['dataset']
    )
    
    # Load and prepare model
    model = _load_model(config, num_features, num_classes)

    # Prepare dataset
    train_data, whole_data = prepare_dataset(
        config['pipeline_params'], 
        data, 
        remove_edge_index=False
    )
    del data
    logger.info(f'Training data: {train_data}')
    logger.info(f'Whole data: {whole_data}')

    # Evaluate model before editing
    bef_edit_results = test(model, whole_data)
    bef_edit_train_acc, bef_edit_valid_acc, bef_edit_test_acc = bef_edit_results
    logger.info(
        f'Before edit - Train acc: {bef_edit_train_acc:.4f}, '
        f'Valid acc: {bef_edit_valid_acc:.4f}, '
        f'Test acc: {bef_edit_test_acc:.4f}'
    )

    # Select target nodes for editing
    node_idx_2flip, flipped_label = select_edit_target_nodes(
        model=model,
        whole_data=whole_data,
        num_samples=config['eval_params']['num_targets'],
        from_valid_set=True
    )
    node_idx_2flip, flipped_label = node_idx_2flip.cuda(), flipped_label.cuda()

    # Optional fine-tuning for MLP models
    if '_MLP' in config['pipeline_params']['model_name']:
        bef_edit_ft_results = finetune_gnn_mlp(config, model, whole_data, train_data)
        logger.info(f'Fine-tuning results: {bef_edit_ft_results}')

    # Run the selected editing method
    raw_results = _run_edit_method(
        config=config,
        model=model,
        node_idx_2flip=node_idx_2flip,
        flipped_label=flipped_label,
        whole_data=whole_data
    )

    # Process and analyze results
    processed_raw_results = process_edit_results(bef_edit_results, raw_results)
    processed_results = process_raw_exp_results(processed_raw_results)

    return processed_raw_results, processed_results