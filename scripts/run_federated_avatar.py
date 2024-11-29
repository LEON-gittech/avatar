'''
Author: LEON leon.kepler@bytedance.com
Date: 2024-11-29 14:30:33
LastEditors: LEON leon.kepler@bytedance.com
LastEditTime: 2024-11-29 22:20:52
FilePath: /ava/scripts/run_federated_avatar.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import os.path as osp
from avatar.fl.federated_avatar import FederatedAvaTaR
from avatar.kb.flickr30k_entities import Flickr30kEntities
from avatar.models.avatar import AvaTaR
from stark_qa import load_qa, load_skb
from avatar.qa_datasets.dataset import QADataset
from scripts.args import parse_args_w_defaults
import stark_qa

def main():
    # Get arguments with defaults from config file
    args = parse_args_w_defaults('config/default_args.json')
    
    # Load dataset and knowledge base
    kb = load_skb(args.dataset)
    qa_dataset = load_qa(args.dataset)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset in ['amazon', 'mag', 'prime']:
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.query_emb_dir = osp.join(emb_root, 'query')
        args.node_emb_dir = osp.join(emb_root, 'doc')
        args.chunk_emb_dir = osp.join(emb_root, 'chunk')
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)
        os.makedirs(args.chunk_emb_dir, exist_ok=True)

        kb = stark_qa.load_skb(args.dataset)
        qa_dataset = stark_qa.load_qa(args.dataset)

    elif  args.dataset == 'flickr30k_entities':
        emb_root = osp.join(args.emb_dir, args.dataset, args.emb_model)
        args.chunk_emb_dir = None
        args.query_emb_dir = osp.join(emb_root, 'query')
        args.node_emb_dir = osp.join(emb_root, 'image')
        os.makedirs(args.query_emb_dir, exist_ok=True)
        os.makedirs(args.node_emb_dir, exist_ok=True)

        kb = Flickr30kEntities(root=args.root_dir)
        qa_dataset = QADataset(name=args.dataset, root=args.root_dir)
        
    output_dir = osp.join(args.output_dir, 'agent', args.dataset, 'avatar', args.agent_llm)
    os.makedirs(name=output_dir, exist_ok=True)
    # Create base AvaTaR model
    base_model = AvaTaR(
        kb=kb,
        emb_model=args.emb_model,
        agent_llm=args.agent_llm,
        api_func_llm=args.api_func_llm,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        node_emb_dir=args.node_emb_dir,
        query_emb_dir=args.query_emb_dir,
        chunk_emb_dir=args.chunk_emb_dir,
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