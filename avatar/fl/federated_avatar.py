'''
Author: LEON leon.kepler@bytedance.com
Date: 2024-11-29 14:30:33
LastEditors: LEON leon.kepler@bytedance.com
LastEditTime: 2024-11-29 15:14:54
FilePath: /ava/avatar/avatar/fl/federated_avatar.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os.path as osp
from typing import List, Dict, Any
import os

from avatar.models.avatar import AvaTaR
from avatar.fl.client import FederatedClient
from avatar.fl.server import FederatedServer

class FederatedAvaTaR:
    """Implements federated learning for prompt optimization across multiple clients"""
    
    def __init__(self, num_clients: int, base_model: AvaTaR, output_dir: str, aggregate: str = 'llm'):
        """
        Initialize federated AvaTaR optimization.
        
        Args:
            num_clients: Number of federated learning clients
            base_model: Base AvaTaR model
            output_dir: Output directory path
            aggregate: Method to aggregate prompts ('llm' or 'vote')
        """
        self.num_clients = num_clients
        self.output_dir = output_dir
        self.base_model = base_model
        self.aggregate = aggregate
        
        # Initialize server
        self.server = FederatedServer(output_dir, aggregate_method=aggregate)
        
        # Initialize clients
        self.clients = []
        for i in range(num_clients):
            client_dir = osp.join(output_dir, f'client_{i}')
            client = FederatedClient(i, base_model, client_dir)
            self.clients.append(client)

    def run_federated_optimization(self,
                                 qa_dataset: Any,
                                 client_indices: List[List[int]],
                                 n_rounds: int = 5,
                                 **kwargs) -> Dict:
        """
        Run federated optimization process.
        
        Args:
            qa_dataset: QA dataset
            client_indices: List of training indices for each client
            n_rounds: Number of federation rounds
            **kwargs: Additional arguments for client optimization
            
        Returns:
            Dict containing final optimized actions and parameters
        """
        for round_idx in range(n_rounds):
            print(f"Starting federation round {round_idx + 1}/{n_rounds}")
            
            # Client optimization phase
            client_results = []
            for client_id, indices in enumerate(client_indices):
                print(f"Optimizing client {client_id + 1}/{self.num_clients}")
                client_result = self.clients[client_id].optimize(
                    qa_dataset=qa_dataset,
                    train_indices=indices,
                    **kwargs
                )
                client_results.append(client_result)

            # Server aggregation phase
            aggregated_result = self.server.aggregate_prompts(
                client_results,
                sel_metric=kwargs.get('sel_metric', 'MRR')
            )
            
            # Update clients with aggregated results
            for client in self.clients:
                client.update_model(
                    aggregated_result['actions'],
                    aggregated_result['parameters']
                )

        return self.server.best_global_result 