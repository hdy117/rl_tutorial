#!/usr/bin/env python3
"""
合并 RL_Tutorial.md 及其索引的章节文件为一个完整的文档
"""

import os
import re
from pathlib import Path


def extract_chapter_files(index_content):
    """从索引文件中提取章节文件列表（按顺序）"""
    chapters = []
    
    # 匹配表格中的文件链接: | 第X章 | 标题 | [文件名](路径) |
    pattern = r'\|\s*第(\d+)章\s*\|[^|]+\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|'
    matches = re.findall(pattern, index_content)
    
    for chapter_num, filename, filepath in matches:
        chapters.append({
            'num': int(chapter_num),
            'filename': filename,
            'path': filepath.strip()
        })
    
    # 按章节号排序
    chapters.sort(key=lambda x: x['num'])
    return chapters


def read_file(filepath):
    """读取文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"警告: 无法读取文件 {filepath}: {e}")
        return None


def merge_tutorial():
    """主合并函数"""
    base_dir = Path(__file__).parent
    index_file = base_dir / 'RL_Tutorial.md'
    output_file = base_dir / 'RL_Tutorial_Full.md'
    
    # 读取索引文件
    print(f"读取索引文件: {index_file}")
    index_content = read_file(index_file)
    if not index_content:
        print("错误: 无法读取索引文件")
        return
    
    # 提取章节列表
    chapters = extract_chapter_files(index_content)
    print(f"发现 {len(chapters)} 个章节")
    
    # 开始构建完整文档
    full_content = []
    
    # 添加文档标题和说明
    full_content.append("# 强化学习教程 (RL Tutorial) - 完整版")
    full_content.append("")
    full_content.append("> 本文档由 `RL_Tutorial.md` 及各章节文件自动合并生成")
    full_content.append("> 生成时间: 见文件修改时间")
    full_content.append("")
    full_content.append("---")
    full_content.append("")
    
    # 提取并添加全局导航图和章节目录（去掉"文件"列的表格）
    # 保留导航图
    nav_match = re.search(r'## 全局导航图.*?(?=## 章节目录)', index_content, re.DOTALL)
    if nav_match:
        full_content.append("## 全局导航图")
        full_content.append("")
        # 提取代码块内容
        nav_content = nav_match.group(0)
        code_match = re.search(r'```text.*?```', nav_content, re.DOTALL)
        if code_match:
            full_content.append(code_match.group(0))
        full_content.append("")
    
    # 添加简化版章节目录（去掉文件路径列）
    full_content.append("## 章节目录")
    full_content.append("")
    full_content.append("| 章节 | 标题 |")
    full_content.append("|------|------|")
    
    for ch in chapters:
        # 从索引中提取标题
        pattern = rf'\|\s*第{ch["num"]}章\s*\|\s*([^|]+)\|'
        match = re.search(pattern, index_content)
        if match:
            title = match.group(1).strip()
            full_content.append(f'| 第{ch["num"]}章 | {title} |')
    
    full_content.append("")
    full_content.append("---")
    full_content.append("")
    
    # 逐个添加章节内容
    for i, ch in enumerate(chapters, 1):
        chapter_path = base_dir / ch['path']
        print(f"  [{i}/{len(chapters)}] 处理: {ch['filename']}")
        
        chapter_content = read_file(chapter_path)
        if chapter_content:
            full_content.append("")
            full_content.append("---")
            full_content.append("")
            full_content.append(chapter_content)
        else:
            full_content.append("")
            full_content.append(f"<!-- 章节 {ch['num']} 内容缺失: {ch['filename']} -->")
    
    # 添加学习路径建议
    learning_section = re.search(r'## 学习路径建议.*', index_content, re.DOTALL)
    if learning_section:
        full_content.append("")
        full_content.append("---")
        full_content.append("")
        full_content.append(learning_section.group(0))
    
    # 写入输出文件
    print(f"\n写入合并文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_content))
    
    print(f"✅ 合并完成! 共 {len(chapters)} 个章节")
    print(f"   输出文件: {output_file}")
    print(f"   文件大小: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == '__main__':
    merge_tutorial()
