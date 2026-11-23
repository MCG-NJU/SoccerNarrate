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



def should_keep_segment(segment: Dict, category_filter_mode: str = "1") -> bool:
    """
    根据 category 字段判断是否保留该 segment
    
    Args:
        segment: 包含 category 字段的字典
        category_filter_mode: 过滤模式
            - "1": 只保留 category="1"，删除 "2" 和 "3"
            - "1+2": 保留 category="1" 和 "2"，删除 "3"
            - "all": 保留所有（不过滤）
        
    Returns:
        True 如果应该保留，False 如果应该删除
    """
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
        segment: 包含 start, end, corrected_text, category 的字典
        
    Returns:
        text_stream: [[开始时间, 结束时间, 词语, 类别], ...] 格式的列表
    """
    start_time = segment['start']
    end_time = segment['end']
    text = segment.get('corrected_text_numbered', segment.get('original_text', ''))
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
    match = re.search(r'([12])-224p_processed\.json', json_filename)
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
    min_clip_sec: int = 16,
    max_clip_sec: int = 64,
    max_silence_sec: float = 3.0,
    category_filter_mode: str = "1",
    window_sec: int = 24,
    window_overlap_sec: int = 12,
) -> List[Dict]:
    """
    处理单个 JSON 文件，生成多个固定窗口的训练片段
    
    核心思想：
    - 先把整场比赛的解说转成按词的时间戳流 all_words；
    - 在时间轴上用固定窗口 W=window_sec 做滑动，步长为 (W - window_overlap_sec)，
      得到 {C1, C2, ...}，相邻窗口之间有 window_overlap_sec 秒的重叠；
    - 每个窗口内收集与之相交的词，作为一个训练 clip。
    
    Args:
        json_path: JSON 文件路径
        video_base_dir: 视频文件基础目录
        min_clip_sec: 窗口内有效文本最短时长（秒），低于该值的窗口会被丢弃
        max_clip_sec: 兼容旧参数（当前逻辑中不再主动使用）
        max_silence_sec: 兼容旧参数（保留以便后续需要时使用）
        category_filter_mode: category 过滤模式
            - "1": 只保留 category="1"，删除 "2" 和 "3"
            - "1+2": 保留 category="1" 和 "2"，删除 "3"
            - "all": 保留所有（不过滤）
        window_sec: 单个训练窗口长度 W（例如 24 秒）
        window_overlap_sec: 相邻窗口重叠长度 O（0 < O < W，例如 12 秒）
        
    Returns:
        训练片段列表
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
        if not should_keep_segment(segment, category_filter_mode):
            filtered_count += 1
            continue
        word_stream = segment_text_to_word_stream(segment)
        all_words.extend(word_stream)
    
    if filtered_count > 0:
        print(f"  过滤了 {filtered_count}/{len(segments)} 个 segments (category_filter_mode={category_filter_mode})")
    
    if not all_words:
        return []
    
    # 将词语流切成固定窗口：长度 window_sec，步长 step_sec=window_sec-window_overlap_sec
    if window_sec <= 0:
        raise ValueError(f"window_sec 必须为正数，当前为 {window_sec}")
    if not (0 < window_overlap_sec < window_sec):
        raise ValueError(f"window_overlap_sec 必须满足 0 < O < window_sec，当前为 O={window_overlap_sec}, window_sec={window_sec}")
    
    step_sec = window_sec - window_overlap_sec
    
    clips = []
    global_start = all_words[0][0]
    global_end = all_words[-1][1]
    
    # 双指针，用于在滑动窗口中高效收集单词
    n_words = len(all_words)
    start_idx = 0
    
    current_start = global_start
    while current_start + window_sec <= global_end and start_idx < n_words:
        window_start = current_start
        window_end = current_start + window_sec
        
        # 跳过完全在窗口前面的单词
        while start_idx < n_words and all_words[start_idx][1] <= window_start:
            start_idx += 1
        
        # 收集与窗口相交的单词
        clip_words: List[List] = []
        idx = start_idx
        while idx < n_words and all_words[idx][0] < window_end:
            word_start, word_end, word_text, word_cat = all_words[idx]
            if word_end > window_start:
                clip_words.append([word_start, word_end, word_text, word_cat])
            idx += 1
        
        # 计算窗口内文本实际覆盖的时长
        if clip_words:
            text_start = clip_words[0][0]
            text_end = clip_words[-1][1]
            text_duration = text_end - text_start
        else:
            text_duration = 0.0
        
        # 要求窗口内至少有一定时长的文本，否则跳过该窗口
        if clip_words and text_duration >= min_clip_sec:
            clips.append({
                'video': video_path,
                'content': clip_words,
                'title': title,
                'category': category,
                'video_start': window_start,
                'video_end': window_end,
            })
        
        current_start += step_sec
    
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
    
    for clip in clips:
        if not clip['content']:
            continue
        
        # 选择 query
        if use_random_query and query_candidates:
            selected_query = random.choice(query_candidates)
        else:
            selected_query = query if query else random.choice(SOCCER_COMMENTARY_QUERIES)
        
        # 获取视频片段的时间范围
        # 如果 clip 中显式给出 video_start / video_end（固定窗口），优先使用；
        # 否则退回到内容本身的起止时间。
        if 'video_start' in clip and 'video_end' in clip:
            video_start = clip['video_start']
            video_end = clip['video_end']
        else:
            video_start = clip['content'][0][0]
            video_end = clip['content'][-1][1]
        
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
                        'category': clip.get('category', '')
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
    window_sec: int = 24,
    window_overlap_sec: int = 12,
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
                if file.endswith('_processed.json'):
                    json_files.append(os.path.join(root, file))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 处理所有文件
    all_conversations = []
    
    for json_path in tqdm(json_files, desc=f"处理 {split} 数据"):
        clips = process_json_file(
            json_path,
            video_base_dir=video_base_dir,
            min_clip_sec=min_clip_sec,
            max_clip_sec=max_clip_sec,
            max_silence_sec=max_silence_sec,
            category_filter_mode=category_filter_mode,
            window_sec=window_sec,
            window_overlap_sec=window_overlap_sec,
        )
        
        conversations = clips_to_conversations(
            clips, 
            query=query, 
            use_random_query=use_random_query
        )
        all_conversations.extend(conversations)
    
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
    parser.add_argument('--max_clip_sec', type=int, default=64,
                       help='最大片段长度（秒）')
    parser.add_argument('--max_silence_sec', type=float, default=5.0,
                       help='最大静音间隔（秒）')
    parser.add_argument('--window_sec', type=int, default=24,
                       help='单个训练窗口长度 W（秒），例如 24')
    parser.add_argument('--window_overlap_sec', type=int, default=12,
                       help='相邻窗口重叠长度 O（秒），满足 0 < O < W，例如 12')
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
            window_sec=args.window_sec,
            window_overlap_sec=args.window_overlap_sec,
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
            category_filter_mode=args.category_filter_mode,
            window_sec=args.window_sec,
            window_overlap_sec=args.window_overlap_sec,
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
            window_sec=args.window_sec,
            window_overlap_sec=args.window_overlap_sec,
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
            category_filter_mode=args.category_filter_mode,
            window_sec=args.window_sec,
            window_overlap_sec=args.window_overlap_sec,
        )

