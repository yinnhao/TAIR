#!/usr/bin/env python3
"""
预处理中文数据集的脚本
"""
import json
import os
import argparse
from collections import Counter
import sys
sys.path.append('.')
from terediff.dataset.chinese_vocab import CTLABELS, is_valid_char

def analyze_dataset(json_path):
    """分析数据集中的字符分布"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_chars = Counter()
    valid_texts = 0
    invalid_texts = 0
    
    for img_id, img_data in data.items():
        text_instances = img_data['0']['text_instances']
        for instance in text_instances:
            text = instance['text']
            
            # 统计字符
            for char in text:
                all_chars[char] += 1
            
            # 检查是否所有字符都支持
            if all(is_valid_char(char) for char in text):
                valid_texts += 1
            else:
                invalid_texts += 1
                unsupported_chars = [char for char in text if not is_valid_char(char)]
                print(f"Unsupported chars in '{text}': {unsupported_chars}")
    
    print(f"Total texts: {valid_texts + invalid_texts}")
    print(f"Valid texts: {valid_texts}")
    print(f"Invalid texts: {invalid_texts}")
    print(f"Coverage: {valid_texts / (valid_texts + invalid_texts) * 100:.2f}%")
    
    # 显示最常见的不支持字符
    unsupported_chars = {char: count for char, count in all_chars.items() 
                        if not is_valid_char(char)}
    if unsupported_chars:
        print("\nTop unsupported characters:")
        for char, count in sorted(unsupported_chars.items(), 
                                key=lambda x: x[1], reverse=True)[:20]:
            print(f"  '{char}': {count}")

def filter_dataset(input_json, output_json):
    """过滤数据集，只保留支持的文本"""
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = {}
    
    for img_id, img_data in data.items():
        text_instances = img_data['0']['text_instances']
        filtered_instances = []
        
        for instance in text_instances:
            text = instance['text']
            if all(is_valid_char(char) for char in text) and len(text) <= 25:
                filtered_instances.append(instance)
        
        if filtered_instances:  # 只保留有有效文本的图片
            filtered_data[img_id] = {
                '0': {'text_instances': filtered_instances}
            }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Filtered dataset saved to {output_json}")
    print(f"Original images: {len(data)}")
    print(f"Filtered images: {len(filtered_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Output filtered JSON file")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't filter")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.input)
    else:
        if not args.output:
            args.output = args.input.replace('.json', '_filtered.json')
        filter_dataset(args.input, args.output)