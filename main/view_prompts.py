#!/usr/bin/env python3
"""
查看训练数据集中的提示词
"""
import pickle
import argparse
import random

def view_prompts(pkl_path, num_samples=20, random_sample=False):
    """查看pickle文件中的提示词"""
    print(f"Loading prompts from: {pkl_path}")
    
    # 加载提示词
    with open(pkl_path, 'rb') as f:
        prompts = pickle.load(f)
    
    print(f"\nTotal number of prompts: {len(prompts)}")
    print("=" * 80)
    
    # 选择要显示的提示词
    if random_sample:
        if num_samples > len(prompts):
            num_samples = len(prompts)
        indices = random.sample(range(len(prompts)), num_samples)
        print(f"\nShowing {num_samples} random samples:")
    else:
        indices = list(range(min(num_samples, len(prompts))))
        print(f"\nShowing first {len(indices)} prompts:")
    
    print("=" * 80)
    
    # 显示提示词
    for i, idx in enumerate(indices, 1):
        prompt = prompts[idx]
        print(f"\n[{i}] Index {idx}:")
        print(f"    {prompt}")
    
    print("\n" + "=" * 80)
    
    # 统计信息
    prompt_lengths = [len(str(p)) if p is not None else 0 for p in prompts]
    avg_length = sum(prompt_lengths) / len(prompt_lengths)
    max_length = max(prompt_lengths)
    min_length = min(prompt_lengths)
    
    print(f"\nStatistics:")
    print(f"  Average prompt length: {avg_length:.1f} characters")
    print(f"  Max prompt length: {max_length} characters")
    print(f"  Min prompt length: {min_length} characters")
    print(f"  None/Empty prompts: {sum(1 for p in prompts if p is None or p == '')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View prompts from pickle file")
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to the pickle file containing prompts")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to display (default: 20)")
    parser.add_argument("--random", action="store_true",
                        help="Sample randomly instead of showing first N")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    
    args = parser.parse_args()
    
    if args.random:
        random.seed(args.seed)
    
    view_prompts(args.pkl_path, args.num_samples, args.random)
