import os
import os.path as osp
import copy
from typing import Dict, List, Any

from avatar.models.avatar import AvaTaR
from stark_qa.tools.io import read_from_file, write_to_file

class FederatedClient:
    """Base class for federated learning clients"""
    
    def __init__(self,
                 client_id: int,
                 base_model: AvaTaR,
                 output_dir: str):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: ID of this client
            base_model: Base AvaTaR model to copy
            output_dir: Output directory for this client
        """
        self.client_id = client_id
        self.model = copy.deepcopy(base_model)
        self.model.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def optimize(self,
                qa_dataset: Any,
                train_indices: List[int],
                n_examples: int = 25,
                n_total_steps: int = 200,
                n_eval: int = 100,
                batch_size: int = 20,
                topk_eval: int = 30,
                patience: int = 10,
                metrics: List[str] = ['hit@5', 'recall@20'],
                sel_metric: str = 'MRR') -> Dict:
        """
        Run local optimization on client's data.
        
        Args:
            qa_dataset: QA dataset
            train_indices: Training indices for this client
            n_examples: Number of examples to use
            n_total_steps: Total optimization steps
            n_eval: Number of examples for evaluation
            batch_size: Batch size for training
            topk_eval: Top k for evaluation
            patience: Patience for early stopping
            metrics: Metrics to track
            sel_metric: Metric for selection
            
        Returns:
            Dict containing optimized actions and parameters
        """
        # Run optimization on client's data
        self.model.optimize_actions(
            qa_dataset=qa_dataset,
            use_group=False,  # Each client has its own subset
            n_examples=n_examples,
            n_total_steps=n_total_steps,
            n_eval=n_eval,
            batch_size=batch_size,
            topk_eval=topk_eval,
            patience=patience,
            metrics=metrics,
            sel_metric=sel_metric
        )

        # Load best actions and parameters
        actions_best_path = osp.join(self.model.output_dir, 'actions_best.txt')
        param_best_path = osp.join(self.model.output_dir, 'actions_best_param.json')
        
        actions_best = read_from_file(actions_best_path)
        param_best = read_from_file(param_best_path)

        return {
            'actions': actions_best,
            'parameters': param_best,
            'client_id': self.client_id
        }

    def update_model(self, actions: str, parameters: Dict) -> None:
        """
        Update client model with aggregated actions and parameters.
        
        Args:
            actions: Aggregated actions string
            parameters: Aggregated parameters dict
        """
        # Save new actions and parameters
        write_to_file(
            osp.join(self.model.output_dir, 'actions_curr.txt'),
            actions
        )
        write_to_file(
            osp.join(self.model.output_dir, 'parameters_curr.json'),
            parameters
        ) 