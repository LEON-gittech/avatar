'''
Author: LEON leon.kepler@bytedance.com
Date: 2024-11-29 14:30:33
LastEditors: LEON leon.kepler@bytedance.com
LastEditTime: 2024-11-29 16:23:39
FilePath: /ava/scripts/run_federated_avatar.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import os.path as osp
from avatar.fl.federated_avatar import FederatedAvaTaR
from avatar.models.avatar import AvaTaR
from stark_qa import load_qa, load_skb
from scripts.args import parse_args

def main():
    # Get arguments
    args = parse_args()
    
    # Load dataset and knowledge base
    kb = load_skb(args.dataset)
    qa_dataset = load_qa(args.dataset)
    print(type(kb))
    print(kb.__dict__)
    print(kb[0])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create base AvaTaR model
    base_model = AvaTaR(
        kb=kb,
        emb_model=args.emb_model,
        agent_llm=args.agent_llm,
        api_func_llm=args.api_func_llm
    )
    
    # Initialize federated optimizer with core parameters
    fed_optimizer = FederatedAvaTaR(
        num_clients=args.num_clients,
        base_model=base_model,
        output_dir=args.output_dir,
        aggregate=args.fl_aggregate
    )

    # Run federated optimization
    final_result = fed_optimizer.run_federated_optimization(
        qa_dataset=qa_dataset,
        split_type=args.split_type,
        n_rounds=args.n_rounds,
        n_examples=args.n_examples,
        n_total_steps=args.n_total_steps,
        n_eval=args.n_eval,
        batch_size=args.batch_size,
        topk_eval=args.topk_eval,
        patience=args.patience,
        metrics=args.metrics,
        sel_metric=args.sel_metric
    )

    # Save final results
    with open(osp.join(args.output_dir, 'final_actions.txt'), 'w') as f:
        f.write(final_result['actions'])

if __name__ == '__main__':
    main() 