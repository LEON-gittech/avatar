"""
Script to inspect and analyze the STaRK and Flickr30k Entities datasets.
"""

import os
import sys
import json
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm
from stark_qa import load_qa, load_skb

def analyze_kb(kb: Any) -> Dict[str, Any]:
    """Analyze knowledge base structure and contents."""
    stats = {
        "node_types": kb.node_type_lst(),
        "edge_types": kb.rel_type_lst(),
        "num_nodes": len(kb.candidate_ids),
        "node_attributes": kb.node_attr_dict
    }
    
    # Sample some nodes for inspection
    sample_nodes = []
    for node_id in kb.candidate_ids[:5]:
        node_info = kb.get_doc_info(node_id)
        sample_nodes.append({
            "id": node_id,
            "info": node_info
        })
    stats["sample_nodes"] = sample_nodes
    
    return stats

def analyze_qa_dataset(qa_dataset: Any) -> Dict[str, Any]:
    """Analyze QA dataset statistics."""
    splits = qa_dataset.get_idx_split()
    stats = {
        "splits": {},
        "query_stats": defaultdict(dict)
    }
    
    for split_name, split_indices in splits.items():
        query_lengths = []
        answer_counts = []
        
        for idx in tqdm(split_indices, desc=f"Analyzing {split_name} split"):
            query, query_id, answer_ids, meta_info = qa_dataset[idx]
            query_lengths.append(len(query.split()))
            answer_counts.append(len(answer_ids))
        
        stats["splits"][split_name] = {
            "size": len(split_indices),
            "avg_query_length": np.mean(query_lengths),
            "avg_answers": np.mean(answer_counts),
            "min_answers": np.min(answer_counts),
            "max_answers": np.max(answer_counts)
        }
        
        # Sample some examples
        sample_examples = []
        for idx in split_indices[:5]:
            query, query_id, answer_ids, meta_info = qa_dataset[idx]
            sample_examples.append({
                "query": query,
                "query_id": query_id,
                "answer_ids": answer_ids,
                "meta_info": meta_info
            })
        stats["splits"][split_name]["samples"] = sample_examples
    
    return stats

def save_analysis(stats: Dict[str, Any], output_file: str):
    """Save analysis results to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

def print_analysis(stats: Dict[str, Any]):
    """Print analysis results in a readable format."""
    print("\n=== Knowledge Base Analysis ===")
    print(f"Node Types: {stats['kb']['node_types']}")
    print(f"Edge Types: {stats['kb']['edge_types']}")
    print(f"Number of Nodes: {stats['kb']['num_nodes']}")
    print("\nNode Attributes:")
    for node_type, attrs in stats['kb']['node_attributes'].items():
        print(f"  {node_type}: {attrs}")
    
    print("\n=== QA Dataset Analysis ===")
    for split_name, split_stats in stats['qa']['splits'].items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Size: {split_stats['size']}")
        print(f"  Average Query Length: {split_stats['avg_query_length']:.2f} words")
        print(f"  Average Answers per Query: {split_stats['avg_answers']:.2f}")
        print(f"  Answer Range: {split_stats['min_answers']} - {split_stats['max_answers']}")
        
        print("\n  Sample Queries:")
        for i, sample in enumerate(split_stats['samples'][:3], 1):
            print(f"    {i}. Query: {sample['query']}")
            print(f"       Answer IDs: {sample['answer_ids']}")

def main():
    # Analyze Amazon dataset
    print("Analyzing Amazon dataset...")
    amazon_kb = load_skb('amazon')
    amazon_qa = load_qa('amazon')
    
    amazon_stats = {
        "kb": analyze_kb(amazon_kb),
        "qa": analyze_qa_dataset(amazon_qa)
    }
    
    # Save results
    save_analysis(amazon_stats, "data/analysis/amazon_analysis.json")
    
    # Print analysis
    print_analysis(amazon_stats)

if __name__ == "__main__":
    main() 