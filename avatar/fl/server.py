import os
import os.path as osp
from typing import List, Dict, Any
import numpy as np

from stark_qa.tools.io import write_to_file
from avatar.fl.utils import get_llm_output

class FederatedServer:
    """Server class for federated learning that handles prompt aggregation using LLM"""
    
    def __init__(self, output_dir: str, aggregate_method: str = "llm"):
        """
        Initialize the federated server.
        
        Args:
            output_dir: Output directory for aggregated results
            aggregate_method: Method to aggregate prompts ('llm' or 'vote')
        """
        self.output_dir = output_dir
        self.aggregate_method = aggregate_method
        os.makedirs(output_dir, exist_ok=True)
        self.best_global_result = None
        self.current_round = 0

    def aggregate_prompts(self, client_results: List[Dict], sel_metric: str = 'MRR') -> Dict:
        """
        Aggregate prompts from clients.
        
        Args:
            client_results: List of client optimization results
            sel_metric: Metric for selecting best actions
            
        Returns:
            Dict containing aggregated actions
        """
        if self.aggregate_method == 'llm':
            return self._aggregate_prompts_llm(client_results, sel_metric)
        elif self.aggregate_method == 'vote':
            return self._aggregate_prompts_vote(client_results, sel_metric)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregate_method}")

    def _aggregate_prompts_llm(self, client_results: List[Dict], sel_metric: str) -> Dict:
        """
        Aggregate prompts from clients using LLM.
        
        Args:
            client_results: List of client optimization results
            sel_metric: Metric for selecting best actions
            
        Returns:
            Dict containing aggregated actions
        """
        # Prepare prompts with metrics for merging
        prompts_with_metrics = []
        for result in client_results:
            prompt_info = {
                'actions': result['actions'],
                'metrics': result.get('metrics', {})
            }
            prompts_with_metrics.append(prompt_info)

        # Merge prompts using LLM
        merged_actions = self._merge_prompts_with_llm(prompts_with_metrics)

        # Save aggregated results for this round
        round_dir = osp.join(self.output_dir, f'round_{self.current_round}')
        os.makedirs(round_dir, exist_ok=True)
        
        write_to_file(
            osp.join(round_dir, 'aggregated_actions.txt'),
            merged_actions
        )

        # Calculate metrics for aggregated result
        metrics = {}
        for metric in set().union(*[r.get('metrics', {}).keys() for r in client_results]):
            values = [r.get('metrics', {}).get(metric, 0) for r in client_results]
            metrics[metric] = np.mean(values)

        # Update best global result if needed
        aggregated_result = {
            'actions': merged_actions,
            'metrics': metrics
        }
        
        if (self.best_global_result is None or 
            aggregated_result['metrics'].get(sel_metric.lower(), 0) > 
            self.best_global_result['metrics'].get(sel_metric.lower(), 0)):
            self.best_global_result = aggregated_result
            
        self.current_round += 1
        return aggregated_result

    def _aggregate_prompts_vote(self, client_results: List[Dict], sel_metric: str) -> Dict:
        """
        Aggregate prompts from clients using voting.
        
        Args:
            client_results: List of client optimization results
            sel_metric: Metric for selecting best actions
            
        Returns:
            Dict containing aggregated actions
        """
        # Select best performing prompt based on voting
        metrics = [r.get('metrics', {}).get(sel_metric.lower(), 0) for r in client_results]
        best_idx = np.argmax(metrics)
        best_actions = client_results[best_idx]['actions']
        
        # Save and return results
        round_dir = osp.join(self.output_dir, f'round_{self.current_round}')
        os.makedirs(round_dir, exist_ok=True)
        
        write_to_file(
            osp.join(round_dir, 'aggregated_actions.txt'),
            best_actions
        )

        aggregated_result = {
            'actions': best_actions,
            'metrics': {sel_metric.lower(): metrics[best_idx]}
        }
        
        if (self.best_global_result is None or 
            aggregated_result['metrics'][sel_metric.lower()] > 
            self.best_global_result['metrics'][sel_metric.lower()]):
            self.best_global_result = aggregated_result
            
        self.current_round += 1
        return aggregated_result

    def _merge_prompts_with_llm(self, client_prompts: List[Dict]) -> str:
        """
        Merge prompts from different clients using LLM.
        
        Args:
            client_prompts: List of client prompts with their metrics
            
        Returns:
            Merged prompt string
        """
        merge_prompt = """You are an expert at merging and improving Python code. You have multiple versions of code that implement the same functionality but with different approaches. Your task is to create a merged version that combines the best aspects of each implementation.

Here are the different code versions with their performance metrics:

{code_sections}

Task:
1. Analyze the different implementations and their performance
2. Identify the most effective patterns and approaches from each version
3. Create a merged implementation that:
   - Combines the best aspects of each version
   - Maintains a clean and efficient structure
   - Preserves the core functionality
   - Optimizes for both performance and readability

The merged code should follow this structure:
```python
# Parameter dictionary with merged parameters
parameter_dict = {{
    # Keep parameters from the best performing version
}}

# Any helper functions needed
def helper_function():
    # Implementation

# Main function implementation
def get_node_score_dict(query, candidate_ids, **parameter_dict):
    # Merged implementation
    return node_score_dict
```

Only output the merged code without any explanations."""

        # Format code sections with metrics
        code_sections = []
        for i, prompt in enumerate(client_prompts):
            metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in prompt.get('metrics', {}).items()])
            code_sections.append(f"Version {i+1} (Performance Metrics:\n{metrics_str}\n):\n```python\n{prompt['actions']}\n```\n")
        
        formatted_prompt = merge_prompt.format(code_sections="\n".join(code_sections))

        # Get merged implementation from LLM
        try:
            merged_code = get_llm_output(formatted_prompt, model=self.llm_model)
            # Extract code between ```python and ``` if present
            if "```python" in merged_code and "```" in merged_code:
                merged_code = merged_code.split("```python")[1].split("```")[0].strip()
            return merged_code
        except Exception as e:
            print(f"Error in LLM merging: {e}")
            # Fallback to best performing code
            best_idx = np.argmax([p.get('metrics', {}).get('mrr', 0) for p in client_prompts])
            return client_prompts[best_idx]['actions']