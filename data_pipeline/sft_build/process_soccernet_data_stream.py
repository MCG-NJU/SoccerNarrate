#!/usr/bin/env python3
"""
将 SoccerNet 数据集处理为 LiveCC 训练数据格式
"""

import os
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict


# 足球解说任务的英文 query 候选列表
SOCCER_COMMENTARY_QUERIES = [
    "Commentate on this match.",
    "Describe what's happening in this video.",
    "Provide live commentary for this clip.",
    "What's happening in this football match?",
    "Comment on the action in this video.",
    "Give a commentary of this play.",
    "Describe the events in this clip.",
    "What's going on in this match?",
    "Provide commentary for this moment.",
    "Tell me what's happening here.",
    "Commentate on this action.",
    "Describe the play in this video.",
    "What can you see in this clip?",
    "Give me a live commentary.",
    "What's the action in this video?",
    "Comment on this football moment.",
    "Describe the match events.",
    "What's happening on the pitch?",
    "Provide match commentary.",
    "Commentate on this scene.",
]



def should_keep_segment(
    segment: Dict,
    category_filter_mode: str = "1",
    require_llm_keep: bool = True,
) -> bool:
    """
    根据 category 字段判断是否保留该 segment
    
    Args:
        segment: 包含 category 字段的字典
        category_filter_mode: 过滤模式
            - "1": 只保留 category="1"，删除 "2" 和 "3"
            - "1+2": 保留 category="1" 和 "2"，删除 "3"
            - "all": 保留所有（不过滤）
        require_llm_keep: 若 segment 含 llm_keep 字段，则仅保留 llm_keep=True 的 segment
        
    Returns:
        True 如果应该保留，False 如果应该删除
    """
    if require_llm_keep:
        llm_keep = segment.get("llm_keep", True)
        if isinstance(llm_keep, str):
            llm_keep = llm_keep.strip().lower() not in {"0", "false", "no", "n"}
        if not bool(llm_keep):
            return False

    if category_filter_mode == "all":
        return True
    
    segment_category = segment.get('category', '')
    # 统一转换为字符串格式，支持数字和字符串两种格式
    segment_category = str(segment_category) if segment_category else ''
    
    if category_filter_mode == "1":
        # 只保留 category="1"
        return segment_category == "1"
    elif category_filter_mode == "1+2":
        # 保留 category="1" 和 "2"
        return segment_category in ["1", "2"]
    else:
        # 默认保留所有
        return True


def segment_text_to_word_stream(segment: Dict) -> List[List]:
    """
    将 segment 的文本转换为词级时间戳流
    
    Args:
        segment: 包含 start, end, {corrected_text/llm_polished_text/original_text}, category 的字典
        
    Returns:
        text_stream: [[开始时间, 结束时间, 词语, 类别], ...] 格式的列表
    """
    start_time = segment['start']
    end_time = segment['end']
    # 新格式优先使用 corrected_text，其次 llm_polished_text，最后 original_text
    text = (
        segment.get("corrected_text")
        or segment.get("llm_polished_text")
        or segment.get("original_text", "")
    )
    category = segment.get('category', '1')  # 默认为类别1
    
    # 按空格分割词语（保留所有内容，包括注释标记）
    words = text.strip().split()
    
    if not words:
        return []
    
    # 处理注释标记：将 [数字] 和 [home/away] 这样的标记合并到前面的词语
    # 这些标记不单独分配时间戳，而是与前面的词语共享
    # 支持三种情况：
    #   1. 只有号码：Meisi [24]
    #   2. 只有主客场：Meisi [home]
    #   3. 两者都有：Meisi [24] [home] 或 Meisi [home] [24]
    processed_words = []
    i = 0
    while i < len(words):
        word = words[i]
        # 检查是否是注释标记： [数字] 或 [home/away]
        is_number_annotation = re.match(r'^\[\d+\]$', word)  # 匹配 [24], [7], [10] 等
        is_homeaway_annotation = re.match(r'^\[(home|away)\]$', word)  # 匹配 [home] 或 [away]
        
        if is_number_annotation or is_homeaway_annotation:
            # 如果是注释标记，将其附加到前一个词语（共享时间戳）
            if processed_words:
                processed_words[-1] += ' ' + word
            else:
                # 如果第一个词就是标记（不应该发生，但处理一下）
                processed_words.append(word)
        else:
            # 普通词语，直接添加
            processed_words.append(word)
        i += 1
    
    # 计算时间戳：需要识别 "词语-数字" 模式，"-数字" 部分不占用额外时间
    # 例如 "Kightly-11" 中的 "-11" 不应该增加时间权重
    num_words_with_timestamps = len(processed_words)
    
    if num_words_with_timestamps == 0:
        return []
    
    # 计算每个词语的平均时长
    total_duration = end_time - start_time
    duration_per_word = total_duration / num_words_with_timestamps if num_words_with_timestamps > 0 else 0
    
    # 生成词级时间戳流
    text_stream = []
    current_time = start_time
    
    for word in processed_words:
        word_start = current_time
        word_end = current_time + duration_per_word
        
        # 保留所有标点符号，不进行清理
        # 检查是否包含注释标记（格式： [数字] 或 [home/away]）
        # 支持三种情况：只有号码、只有主客场、两者都有
        annotation_pattern = r'(\s+\[\d+\]|\s+\[(home|away)\])'
        annotations_match = re.findall(annotation_pattern, word)
        
        if annotations_match:
            # 包含注释标记，保留所有内容（包括标点符号和注释标记）
            # 注释标记已经在前面的处理中附加到词语上了，这里直接使用
            clean_word = word
        else:
            # 普通词语，保留所有内容（包括标点符号）
            # 标点符号会保留在词语中，不单独分配时间戳
            clean_word = word
        
        if clean_word:  # 只添加非空词语
            text_stream.append([round(word_start, 2), round(word_end, 2), clean_word, category])
        
        current_time = word_end
    
    # 确保最后一个词的结束时间等于 segment 的结束时间
    if text_stream:
        text_stream[-1][1] = end_time
    
    return text_stream


def find_video_path(
    json_path: str,
    video_base_dir: str = "/path/to/SoccerNet/videos"
) -> str:
    """
    根据 JSON 文件路径找到对应的视频文件路径
    
    Args:
        json_path: JSON 文件的完整路径，如 /path/to/train/league_season/match_name/1-224p_processed.json
        video_base_dir: 视频文件的基础目录
        
    Returns:
        视频文件路径，如果不存在返回 None
    """
    # 从 JSON 路径提取信息
    # 格式: .../train/league_season/match_name/file_name.json
    parts = Path(json_path).parts
    
    # 找到 split (train/valid/test)
    split_idx = -1
    for i, part in enumerate(parts):
        if part in ['train', 'valid', 'test']:
            split_idx = i
            break
    
    if split_idx == -1:
        return None
    
    # 获取 league_season 和 match_name
    # 例如: spain_laliga_2015-2016, 2016-03-20 - 22-30 Real Madrid 4 - 0 Sevilla
    league_season = parts[split_idx + 1] if split_idx + 1 < len(parts) else None
    match_name = parts[split_idx + 2] if split_idx + 2 < len(parts) else None
    
    # 从文件名提取视频编号（1 或 2）
    json_filename = Path(json_path).name
    # 兼容旧格式：1-224p_processed.json
    # 兼容新格式：1-224p_processed_clean.json
    match = re.search(r"([12])-224p_processed(?:_clean)?\.json$", json_filename)
    if not match:
        return None
    
    video_num = match.group(1)
    
    # 构建视频路径
    # 格式: {video_base_dir}/{league}/{season}/{match_name}/{video_num}_224p.mkv
    # 例如: spain_laliga_2015-2016 -> spain_laliga/2015-2016
    league, season = league_season.rsplit('_', 1)  # 从最后的下划线分割
    
    video_filename = f"{video_num}_720p.mkv"
    video_path = os.path.join(video_base_dir, league, season, match_name, video_filename)
    
    # 检查文件是否存在
    if os.path.exists(video_path):
        return video_path
    
    # 如果不存在，尝试 .mp4 格式
    video_path_mp4 = video_path.replace('.mkv', '.mp4')
    if os.path.exists(video_path_mp4):
        return video_path_mp4
    
    return None


def process_json_file(
    json_path: str,
    video_base_dir: str = "/path/to/SoccerNet/videos",
    min_clip_sec: int = 30,
    max_clip_sec: int = 240,
    max_silence_sec: float = 3.0,
    category_filter_mode: str = "1"
) -> List[Dict]:
    """
    处理单个 JSON 文件，生成多个对话样本
    
    Args:
        json_path: JSON 文件路径
        video_base_dir: 视频文件基础目录
        min_clip_sec: 最小片段长度（秒）
        max_clip_sec: 最大片段长度（秒）
        max_silence_sec: 最大静音间隔（秒）
        category_filter_mode: category 过滤模式
            - "1": 只保留 category="1"，删除 "2" 和 "3"
            - "1+2": 保留 category="1" 和 "2"，删除 "3"
            - "all": 保留所有（不过滤）
        
    Returns:
        对话样本列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取视频路径
    video_path = find_video_path(json_path, video_base_dir)
    # if not video_path:
    #     return []
    
    # 获取元数据
    teams = data.get('teams', [])
    title = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else data.get('gameDate', '')
    category = "soccer"
    
    # 处理所有 segments 为词级流（先过滤 category）
    all_words = []
    segments = data.get('segments', [])
    filtered_count = 0
    for segment in segments:
        # 根据 category 过滤
        if not should_keep_segment(segment, category_filter_mode, require_llm_keep=True):
            filtered_count += 1
            continue
        word_stream = segment_text_to_word_stream(segment)
        all_words.extend(word_stream)
    
    if filtered_count > 0:
        print(
            f"  过滤了 {filtered_count}/{len(segments)} 个 segments "
            f"(category_filter_mode={category_filter_mode}, require_llm_keep=True)"
        )
    
    if not all_words:
        return []
    
    # 将词语流分割成符合要求的片段
    clips = []
    i = 0
    
    while i < len(all_words):
        # 第一个词总是可以作为起始点
        clip_start_time = all_words[i][0]
        clip_start_idx = i
        
        # 找到片段结束点
        j = None
        exceeded_max = False
        
        # 尝试找到合适的结束点（在句子边界，或达到最大长度）
        for j in range(i + 1, len(all_words)):
            clip_end_time = all_words[j][1]
            clip_duration = clip_end_time - clip_start_time
            
            # 如果超过了最大长度，停止
            if clip_duration > max_clip_sec:
                exceeded_max = True
                break
            
            # 检查是否有大的时间间隔（静音）
            if j > i and all_words[j][0] - all_words[j-1][1] > max_silence_sec:
                break
            
            # 如果找到句子边界，也考虑作为结束点（但继续检查是否达到最小长度）
            if all_words[j-1][2].endswith(('.', '!', '?')):
                # 如果已经达到最小长度，可以在此结束
                if clip_duration >= min_clip_sec:
                    break
        
        # 确定片段的实际结束点
        if j is not None and j > i:
            if exceeded_max:
                # 如果超过最大长度，使用 j-1 作为结束点
                end_idx = j - 1
            else:
                end_idx = j
            
            # 确保结束点在有效范围内
            if end_idx <= i:
                end_idx = i + 1
            
            # 计算实际片段时长
            final_clip_end_time = all_words[end_idx-1][1]
            clip_duration = final_clip_end_time - clip_start_time
            
            # 如果片段长度符合要求
            if clip_duration >= min_clip_sec:
                clip_words = all_words[clip_start_idx:end_idx]
                
                clips.append({
                    'video': video_path,
                    'content': clip_words,
                    'title': title,
                    'category': category
                })
            
            # 移动到下一个位置（优先在句子边界，否则在 end_idx）
            i = end_idx
        else:
            # 没有找到合适的结束点（到达末尾），检查剩余部分
            if i < len(all_words):
                final_time = all_words[-1][1]
                remaining_duration = final_time - clip_start_time
                
                if remaining_duration >= min_clip_sec:
                    # 剩余部分作为一个完整的片段（如果不超过最大长度，或者强制分割）
                    if remaining_duration <= max_clip_sec:
                        clip_words = all_words[i:]
                        
                        clips.append({
                            'video': video_path,
                            'content': clip_words,
                            'title': title,
                            'category': category
                        })
                    else:
                        # 即使超过最大长度，也要分割（强制在最大长度处分割）
                        # 找到最大长度对应的位置
                        target_time = clip_start_time + max_clip_sec
                        end_idx = i + 1
                        for idx in range(i + 1, len(all_words)):
                            if all_words[idx][1] > target_time:
                                end_idx = idx
                                break
                        
                        if end_idx > i:
                            clip_words = all_words[i:end_idx]
                            
                            clips.append({
                                'video': video_path,
                                'content': clip_words,
                                'title': title,
                                'category': category
                            })
                            i = end_idx
                            continue
                break
            else:
                break
    
    return clips


def clips_to_conversations(
    clips: List[Dict], 
    query: str = None,
    use_random_query: bool = True,
    query_candidates: List[str] = None
) -> List[List]:
    """
    将 clips 转换为对话格式
    
    Args:
        clips: 片段列表
        query: 用户查询文本（如果 use_random_query=False 时使用）
        use_random_query: 是否随机选择 query
        query_candidates: query 候选列表，如果为 None 则使用默认的 SOCCER_COMMENTARY_QUERIES
        
    Returns:
        对话列表
    """
    conversations = []
    
    # 确定使用的 query 候选列表
    if query_candidates is None:
        query_candidates = SOCCER_COMMENTARY_QUERIES
    
    # 只使用之前两段 clip 的文本作为 previous text
    previous_clip_texts: List[str] = []
    
    for clip_idx, clip in enumerate(clips):
        if not clip['content']:
            continue
        
        # 选择 query
        if use_random_query and query_candidates:
            selected_query = random.choice(query_candidates)
        else:
            selected_query = query if query else random.choice(SOCCER_COMMENTARY_QUERIES)
        
        # 获取视频片段的时间范围
        video_start = clip['content'][0][0]
        video_end = clip['content'][-1][1]
        
        # 构建 previous 字段：取前两段 clip 的 text 拼接
        previous_text = " ".join(previous_clip_texts[-2:]).strip()
        
        # 构建对话
        conversation = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'video',
                        'video': clip['video'],
                        'video_start': video_start,
                        'video_end': video_end
                    },
                    {
                        'type': 'text',
                        'text': selected_query,
                        'title': clip.get('title', ''),
                        'category': clip.get('category', ''),
                        'previous': previous_text
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text_stream',
                        'text_stream': clip['content']
                    }
                ]
            }
        ]
        
        conversations.append(conversation)
        
        # 将当前 clip 的文本内容添加到缓存中，供下一个 clip 使用
        clip_text = " ".join([item[2] for item in clip["content"]]).strip()
        if clip_text:
            previous_clip_texts.append(clip_text)
    
    return conversations


def process_dataset(
    data_dir: str,
    split: str,
    output_file: str,
    video_base_dir: str = "/path/to/SoccerNet/videos",
    min_clip_sec: int = 30,
    max_clip_sec: int = 240,
    max_silence_sec: float = 3.0,
    query: str = None,
    use_random_query: bool = True,
    category_filter_mode: str = "1",
    splits_to_merge: List[str] = None
):
    """
    处理整个数据集分割（train/valid/test）
    
    Args:
        data_dir: 数据目录（包含 train/valid/test）
        split: 分割名称（train/valid/test），用于显示和输出文件名
        output_file: 输出 JSONL 文件路径
        video_base_dir: 视频文件基础目录
        min_clip_sec: 最小片段长度
        max_clip_sec: 最大片段长度
        max_silence_sec: 最大静音间隔
        query: 用户查询文本（如果 use_random_query=False 时使用）
        use_random_query: 是否随机选择 query
        category_filter_mode: category 过滤模式
            - "1": 只保留 category="1"，删除 "2" 和 "3"
            - "1+2": 保留 category="1" 和 "2"，删除 "3"
            - "all": 保留所有（不过滤）
        splits_to_merge: 要合并的多个 split 列表，如果为 None 则只处理单个 split
    """
    # 如果指定了要合并的 splits，则处理多个目录
    if splits_to_merge:
        split_dirs = [os.path.join(data_dir, s) for s in splits_to_merge]
        print(f"合并处理多个 split: {splits_to_merge}")
    else:
        split_dirs = [os.path.join(data_dir, split)]
    
    # 找到所有 JSON 文件
    json_files = []
    for split_dir in split_dirs:
        if not os.path.exists(split_dir):
            print(f"警告: 目录不存在: {split_dir}")
            continue
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                # 兼容旧格式：*_processed.json
                # 兼容新格式：*_processed_clean.json
                if file.endswith("_processed.json") or file.endswith("_processed_clean.json"):
                    json_files.append(os.path.join(root, file))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 处理所有文件
    all_conversations = []
    
    for json_path in tqdm(json_files, desc=f"处理 {split} 数据"):
        try:
            clips = process_json_file(
                json_path,
                video_base_dir=video_base_dir,
                min_clip_sec=min_clip_sec,
                max_clip_sec=max_clip_sec,
                max_silence_sec=max_silence_sec,
                category_filter_mode=category_filter_mode
            )
            
            conversations = clips_to_conversations(
                clips, 
                query=query, 
                use_random_query=use_random_query
            )
            all_conversations.extend(conversations)
            
        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")
            continue
    
    print(f"共生成 {len(all_conversations)} 个对话样本")
    
    # 写入 JSONL 文件并生成 seek 索引
    print(f"写入文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        lengths = []
        
        for conversation in tqdm(all_conversations, desc="写入数据"):
            line = json.dumps(conversation, ensure_ascii=False) + '\n'
            line_bytes = line.encode('utf-8')
            lengths.append(len(line_bytes))
            f.write(line)
        
        # 生成 seek 索引（参考 to_conversation.py 的逻辑）
        # 如果有 N 个数据样本，应该有 N 个 seek 索引
        # seeks[0] = 0 (第一个样本的开始)
        # seeks[1] = lengths[0] (第二个样本的开始)
        # seeks[2] = lengths[0] + lengths[1] (第三个样本的开始)
        # ...
        seeks = [0]
        current = 0
        for length in lengths[:-1]:  # 只处理前 N-1 个长度
            current += length
            seeks.append(current)
        
        # 写入 seek 索引（最后一行）
        f.write(json.dumps(seeks) + '\n')
    
    print(f"完成！共写入 {len(all_conversations)} 个样本")
    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='处理 SoccerNet 数据集为 LiveCC 格式')
    parser.add_argument('--data_dir', type=str, 
                       default='/path/to/SoccerNet/asr/annos',
                       help='数据目录（包含 train/valid/test）')
    parser.add_argument('--video_base_dir', type=str,
                       default='/path/to/SoccerNet/videos',
                       help='视频文件基础目录')
    parser.add_argument('--output_dir', type=str,
                       default='/path/to/SoccerNet/asr/annos/12',
                       help='输出目录')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'test', 'all'],
                       default='all', help='要处理的拆分')
    parser.add_argument('--min_clip_sec', type=int, default=16,
                       help='最小片段长度（秒）')
    parser.add_argument('--max_clip_sec', type=int, default=32,
                       help='最大片段长度（秒）')
    parser.add_argument('--max_silence_sec', type=float, default=10.0,
                       help='最大静音间隔（秒）')
    parser.add_argument('--query', type=str, default=None,
                       help='用户查询文本（如果指定，则 use_random_query 自动设为 False）')
    parser.add_argument('--use_random_query', default=True,
                       help='是否随机选择 query（默认 True）')
    parser.add_argument('--no_random_query', dest='use_random_query', action='store_false',
                       help='不使用随机 query，使用固定的 --query 参数')
    parser.add_argument('--category_filter_mode', type=str, 
                       choices=['1', '1+2', 'all'],
                       default='1+2',
                       help='Category 过滤模式: "1"=只保留category=1, "1+2"=保留category=1和2, "all"=保留所有')
    
    args = parser.parse_args()
    
    # 如果指定了 query，则自动禁用随机选择
    if args.query is not None:
        args.use_random_query = False
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理指定的拆分
    # 将 train 和 valid 合并为 train
    if args.split == 'all':
        # 处理 train（合并 train 和 valid）
        output_file = os.path.join(args.output_dir, f'soccernet_train_with_seeks.jsonl')
        print(f"\n{'='*60}")
        print(f"处理 train 数据（合并 train 和 valid）")
        print(f"{'='*60}")
        process_dataset(
            data_dir=args.data_dir,
            split='train',
            output_file=output_file,
            video_base_dir=args.video_base_dir,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            max_silence_sec=args.max_silence_sec,
            query=args.query,
            use_random_query=args.use_random_query,
            category_filter_mode=args.category_filter_mode,
            splits_to_merge=['train', 'valid']
        )
        
        # 处理 test
        output_file = os.path.join(args.output_dir, f'soccernet_test_with_seeks.jsonl')
        print(f"\n{'='*60}")
        print(f"处理 test 数据")
        print(f"{'='*60}")
        process_dataset(
            data_dir=args.data_dir,
            split='test',
            output_file=output_file,
            video_base_dir=args.video_base_dir,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            max_silence_sec=args.max_silence_sec,
            query=args.query,
            use_random_query=args.use_random_query,
            category_filter_mode=args.category_filter_mode
        )
    elif args.split == 'train':
        # 处理 train（合并 train 和 valid）
        output_file = os.path.join(args.output_dir, f'soccernet_train_with_seeks.jsonl')
        print(f"\n{'='*60}")
        print(f"处理 train 数据（合并 train 和 valid）")
        print(f"{'='*60}")
        process_dataset(
            data_dir=args.data_dir,
            split='train',
            output_file=output_file,
            video_base_dir=args.video_base_dir,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            max_silence_sec=args.max_silence_sec,
            query=args.query,
            use_random_query=args.use_random_query,
            category_filter_mode=args.category_filter_mode,
            splits_to_merge=['train', 'valid']
        )
    else:
        # 处理其他 split（valid 或 test）
        output_file = os.path.join(args.output_dir, f'soccernet_{args.split}_with_seeks.jsonl')
        print(f"\n{'='*60}")
        print(f"处理 {args.split} 数据")
        print(f"{'='*60}")
        process_dataset(
            data_dir=args.data_dir,
            split=args.split,
            output_file=output_file,
            video_base_dir=args.video_base_dir,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            max_silence_sec=args.max_silence_sec,
            query=args.query,
            use_random_query=args.use_random_query,
            category_filter_mode=args.category_filter_mode
        )
