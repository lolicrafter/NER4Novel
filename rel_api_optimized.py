# encoding=utf-8
# author： s0mE
# subject： 使用在线 API 进行人物关系提取（优化版：两阶段策略）
# date： 2024
import argparse
import os
import re
import json
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib
# 在 CI 环境中使用非交互式后端
if os.getenv('CI') == 'true' or os.getenv('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体
import matplotlib.font_manager as fm

chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                 'Noto Sans CJK SC', 'Source Han Sans CN', 'Droid Sans Fallback', 'DejaVu Sans']
available_fonts = [f.name for f in fm.fontManager.ttflist]

font_found = None
for font in chinese_fonts:
    if font in available_fonts:
        font_found = font
        break

if font_found:
    plt.rcParams["font.sans-serif"] = [font_found] + plt.rcParams["font.sans-serif"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams["font.sans-serif"]
    print("⚠️ Warning: No Chinese font found, Chinese characters may display as squares")

plt.rcParams["axes.unicode_minus"] = False

# 尝试导入 rel.py 中的人名识别模块
try:
    # 导入 rel.py 中的 hanlp 类和 count_names 函数
    import sys
    import importlib.util
    
    # 动态导入 rel.py 模块
    spec = importlib.util.spec_from_file_location("rel_module", "rel.py")
    rel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rel_module)
    
    HANLP_AVAILABLE = True
    print("✅ 已加载 rel.py 中的人名识别模块")
except Exception as e:
    HANLP_AVAILABLE = False
    print(f"⚠️ 无法加载 rel.py 模块，将使用简单的人名识别方法: {e}")


class LLMAPI:
    """统一的 LLM API 接口（OpenAI 格式）"""
    
    def __init__(self, base_url, api_key, model_name):
        """
        Args:
            base_url: API 基础 URL（如 https://miaodi.zeabur.app）
            api_key: API 密钥
            model_name: 模型名称（如 deepseek-ai/DeepSeek-V3-0324）
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("需要设置 API 密钥，通过参数或环境变量 API_KEY")
        
        # 检查 base_url 是否包含 /v1/chat/completions
        if '/v1/chat/completions' in self.base_url:
            self.endpoint = self.base_url
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"
    
    def call_api(self, prompt, max_tokens=2000, temperature=0.3):
        """调用 API"""
        import requests
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            response = requests.post(self.endpoint, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # 解析响应（OpenAI 格式）
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            print(f"❌ API 调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   响应内容: {e.response.text[:500]}")
            raise


def extract_names_with_hanlp(file_path):
    """
    使用 rel.py 中的 HanLP 方法提取人名
    
    Args:
        file_path: 文本文件路径
    
    Returns:
        names_list: 人名列表
        nr_nrf_dict: 人名统计字典
    """
    if not HANLP_AVAILABLE:
        return None, None
    
    try:
        # 使用 rel.py 中的方法
        model = rel_module.hanlp(custom_dict=True)
        _, names, nr_nrf_dict = rel_module.count_names(file_path, model)
        
        # 使用 filter_nr 获取高频可信名称（这是关键步骤）
        try:
            # filter_nr 会根据阈值自动过滤，只返回高频名字
            auto_name_list, _ = rel_module.filter_nr(nr_nrf_dict, threshold=-1, first=False)
            # 优先使用 filter_nr 返回的高频名字列表
            names_list = auto_name_list
            print(f"✅ 使用 filter_nr 过滤后得到 {len(names_list)} 个高频人名")
        except Exception as e:
            # 如果过滤失败，使用原始名字列表（但也要按频率排序）
            print(f"⚠️ filter_nr 失败，使用原始名字列表: {e}")
            names_list = list(names)
            # 按出现频率排序
            if nr_nrf_dict:
                name_counts = {}
                for name in names_list:
                    count = nr_nrf_dict.get("nr", {}).get(name, 0) + nr_nrf_dict.get("nrf", {}).get(name, 0)
                    name_counts[name] = count
                names_list = sorted(names_list, key=lambda x: name_counts.get(x, 0), reverse=True)
        
        print(f"✅ 使用 HanLP 提取到 {len(names_list)} 个人名（高频）")
        return names_list, nr_nrf_dict
    except Exception as e:
        print(f"⚠️ HanLP 提取失败: {e}")
        return None, None


def extract_names_simple(text, min_name_length=2):
    """
    简单的人名识别（基于常见模式）
    在 GitHub Actions 中，如果 HanLP 不可用，使用这个简单方法作为后备
    """
    # 常见的中文姓氏
    surnames = ['张', '王', '李', '赵', '刘', '陈', '杨', '黄', '周', '吴', '徐', '孙', 
                 '马', '朱', '胡', '林', '郭', '何', '高', '罗', '郑', '梁', '谢', '宋',
                 '唐', '许', '韩', '冯', '邓', '曹', '彭', '曾', '肖', '田', '董', '袁',
                 '潘', '于', '蒋', '蔡', '余', '杜', '叶', '程', '苏', '魏', '吕', '丁',
                 '任', '沈', '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
                 '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟', '熊', '秦',
                 '邱', '江', '尹', '薛', '闫', '段', '雷', '侯', '龙', '史', '陶', '黎',
                 '贺', '顾', '毛', '郝', '龚', '邵', '万', '钱', '严', '覃', '武', '戴',
                 '莫', '孔', '向', '汤', '常', '路']
    
    names = set()
    # 查找 2-4 字的中文姓名模式
    # 模式：姓氏 + 1-3个汉字
    pattern = r'([' + ''.join(surnames) + r'][' + '\u4e00-\u9fa5' + r']{1,3})'
    matches = re.findall(pattern, text)
    names.update(matches)
    
    # 也查找连续的中文名字（2-4字）
    pattern2 = r'[\u4e00-\u9fa5]{2,4}'
    matches2 = re.findall(pattern2, text)
    # 过滤掉明显不是名字的词
    exclude_words = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一',
                     '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
                     '没有', '看', '好', '自己', '这个', '那个', '这样', '那样'}
    for match in matches2:
        if match not in exclude_words and len(match) >= min_name_length:
            names.add(match)
    
    return list(names)


def find_sentences_with_two_names(text_lines, names_list, max_sentences=200):
    """
    找出包含至少两个人名的句子
    
    Args:
        text_lines: 文本行列表
        names_list: 人名列表
        max_sentences: 最多返回的句子数
    
    Returns:
        sentences: [(sentence, person1, person2, line_index), ...]
    """
    # 构建人名匹配模式（按长度排序，优先匹配长名字）
    names_sorted = sorted(set(names_list), key=len, reverse=True)
    name_pattern = '|'.join(re.escape(name) for name in names_sorted if len(name) >= 2)
    
    if not name_pattern:
        return []
    
    sentences = []
    sentence_pattern = r'[。！？；\n]+'
    
    for line_idx, line in enumerate(text_lines):
        # 按句子分割
        line_sentences = re.split(sentence_pattern, line)
        
        for sentence in line_sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:  # 跳过太短的句子
                continue
            
            # 找出句子中出现的所有人名
            found_names = []
            for name in names_sorted:
                if name in sentence:
                    found_names.append(name)
            
            # 如果找到至少两个人名，记录下来
            if len(found_names) >= 2:
                # 记录所有可能的人名对
                for i in range(len(found_names)):
                    for j in range(i + 1, len(found_names)):
                        person1, person2 = found_names[i], found_names[j]
                        if person1 != person2:
                            sentences.append((sentence, person1, person2, line_idx))
                            
                            if len(sentences) >= max_sentences:
                                return sentences[:max_sentences]
    
    return sentences


def extract_paragraph_context(text_lines, sentence_line_idx, context_lines=3):
    """
    提取句子所在的段落上下文
    
    Args:
        text_lines: 文本行列表
        sentence_line_idx: 句子所在的行索引
        context_lines: 上下文行数（前后各多少行）
    
    Returns:
        paragraph: 段落文本
    """
    start_idx = max(0, sentence_line_idx - context_lines)
    end_idx = min(len(text_lines), sentence_line_idx + context_lines + 1)
    
    paragraph = '\n'.join(text_lines[start_idx:end_idx])
    return paragraph.strip()


def extract_relationships_optimized(file_path, base_url, api_key, model_name,
                                   max_sentences=200, context_lines=3):
    """
    优化的两阶段关系提取策略
    
    阶段1: 使用简单方法找出包含两个人物的句子
    阶段2: 提取这些句子的段落上下文
    阶段3: 使用 LLM 分析段落中的人物关系
    """
    # 初始化 API
    try:
        llm = LLMAPI(base_url=base_url, api_key=api_key, model_name=model_name)
        print(f"✅ 已连接到 API: {base_url}")
        print(f"📦 使用模型: {model_name}")
    except Exception as e:
        print(f"❌ API 初始化失败: {e}")
        raise
    
    # 读取文本文件
    print("📖 正在读取文本文件...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            lines = f.readlines()
    
    # 清理文本行
    text_lines = [line.strip() for line in lines if line.strip()]
    
    print(f"📝 总共有 {len(text_lines)} 行文本")
    
    # 阶段1: 提取所有人名（优先使用 rel.py 中的 HanLP 方法）
    print("\n🔍 阶段1: 提取文本中的人名...")
    
    # 尝试使用 rel.py 中的 HanLP 方法
    names_list, nr_nrf_dict = extract_names_with_hanlp(file_path)
    
    if names_list is None:
        # 如果 HanLP 不可用，使用简单方法
        print("⚠️ 使用简单的人名识别方法（HanLP 不可用）")
        all_text = '\n'.join(text_lines)
        names_list = extract_names_simple(all_text)
        
        # 过滤和去重
        exclude_words = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一',
                         '这个', '那个', '这样', '那样', '什么', '怎么', '为什么', '可以',
                         '不能', '不会', '没有', '不是', '也要', '还要', '还要', '还要'}
        names_list = [name for name in names_list 
                      if len(name) >= 2 and name not in exclude_words]
        
        # 统计名字出现频率，只保留高频名字
        name_counts = defaultdict(int)
        for name in names_list:
            name_counts[name] += all_text.count(name)
        
        # 保留出现至少3次的名字
        names_list = [name for name in names_list if name_counts[name] >= 3]
        names_list = sorted(set(names_list), key=lambda x: name_counts[x], reverse=True)
    
    # 过滤掉无效名字（只包含标点符号等）
    def is_invalid_name(name):
        if not name or name.strip() == '':
            return True
        if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', name):
            return False
        return True
    
    names_list = [name for name in names_list if not is_invalid_name(name)]
    
    # 限制名字数量（最多50个）
    names_list = names_list[:50]
    
    print(f"✅ 找到 {len(names_list)} 个人名")
    if len(names_list) > 0:
        print(f"   前10个: {names_list[:10]}")
    
    if len(names_list) < 2:
        print("⚠️ 人名太少，无法提取关系")
        return [], []
    
    # 阶段2: 找出包含两个人名的句子
    print(f"\n🔍 阶段2: 找出包含至少两个人名的句子（最多 {max_sentences} 个）...")
    sentences_with_names = find_sentences_with_two_names(
        text_lines, names_list, max_sentences=max_sentences
    )
    
    print(f"✅ 找到 {len(sentences_with_names)} 个包含两个人名的句子")
    
    if len(sentences_with_names) == 0:
        print("⚠️ 未找到包含两个人名的句子")
        return [], names_list
    
    # 阶段3: 提取段落并分析
    print(f"\n🔍 阶段3: 提取段落上下文并使用 LLM 分析关系...")
    
    # 去重：同一个句子可能有多个人名对
    seen_sentences = set()
    unique_sentences = []
    for sentence, p1, p2, line_idx in sentences_with_names:
        sentence_key = (sentence, line_idx)
        if sentence_key not in seen_sentences:
            seen_sentences.add(sentence_key)
            unique_sentences.append((sentence, p1, p2, line_idx))
    
    print(f"✅ 去重后共有 {len(unique_sentences)} 个唯一段落")
    
    relationships = []
    entities = set(names_list)
    
    # 构建提示词模板
    prompt_template = """你是一个专业的小说分析助手。请从以下文本段落中提取人物关系。

要求：
1. 识别段落中出现的所有人物姓名
2. 提取人物之间的关系（如：父子、朋友、恋人、同事、敌人、师生、主仆、兄弟、姐妹等）
3. 如果关系不明确，使用"相关"作为关系类型
4. 只提取明确出现的关系，不要推测

输出格式为 JSON 数组，每个元素格式如下：
{
  "person1": "人物1",
  "relation": "关系类型",
  "person2": "人物2"
}

文本段落：
{text}

请只返回 JSON 数组，不要包含其他解释文字。如果文本中没有人物关系，返回空数组 []。"""

    # 分批处理段落
    batch_size = 5  # 每批处理5个段落
    for i in tqdm(range(0, len(unique_sentences), batch_size), desc="分析段落"):
        batch = unique_sentences[i:i+batch_size]
        
        # 提取每个句子的段落上下文
        paragraphs = []
        for sentence, p1, p2, line_idx in batch:
            paragraph = extract_paragraph_context(text_lines, line_idx, context_lines)
            paragraphs.append(paragraph)
        
        # 合并段落
        combined_text = "\n\n---段落分隔---\n\n".join(paragraphs)
        
        # 调用 LLM
        try:
            prompt = prompt_template.format(text=combined_text[:3000])  # 限制长度
            response = llm.call_api(prompt, max_tokens=2000, temperature=0.3)
            
            # 解析响应
            try:
                # 尝试提取 JSON
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    rel_data = json.loads(json_match.group())
                else:
                    rel_data = json.loads(response)
                
                # 处理提取的关系
                for item in rel_data:
                    if isinstance(item, dict) and 'person1' in item and 'person2' in item:
                        person1 = item.get('person1', '').strip()
                        person2 = item.get('person2', '').strip()
                        relation = item.get('relation', '相关').strip()
                        
                        if person1 and person2 and person1 != person2:
                            entities.add(person1)
                            entities.add(person2)
                            relationships.append((
                                person1,
                                relation,
                                person2,
                                0.8  # 默认置信度
                            ))
                
            except json.JSONDecodeError as e:
                print(f"\n⚠️ JSON 解析失败: {e}")
                print(f"   响应内容: {response[:200]}")
            
            # 避免 API 限流
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n⚠️ 处理批次时出错: {e}")
            continue
    
    print(f"\n✅ 提取完成: 发现 {len(entities)} 个人物，{len(relationships)} 个关系")
    return relationships, list(entities)


# 复用原有的关系图绘制和 Excel 导出函数
def build_relationship_graph(relationships, entities=None):
    """构建人物关系图"""
    G = nx.Graph()
    rel_dict = defaultdict(list)
    
    if entities:
        for entity in entities:
            G.add_node(entity)
    
    for person1, relation, person2, confidence in relationships:
        if person1 and person2 and person1 != person2:
            G.add_node(person1)
            G.add_node(person2)
            G.add_edge(person1, person2, weight=1.0, relation=relation, confidence=confidence)
            rel_dict[(person1, person2)].append((relation, confidence))
    
    return G, dict(rel_dict)


def plot_relationship_graph(G, relationships, save_path=None, book_name=None):
    """绘制人物关系图"""
    if not G.nodes():
        print("⚠️ 图中没有节点，无法绘制")
        return
    
    degrees = dict(G.degree())
    
    if nx.is_connected(G):
        main_G = G
    else:
        components = list(nx.connected_components(G))
        main_component = max(components, key=len)
        main_G = G.subgraph(main_component).copy()
        print(f"📊 主要子图包含 {len(main_component)} 个节点（共 {len(G.nodes())} 个节点）")
    
    node_sizes = [degrees.get(node, 1) * 500 for node in main_G.nodes()]
    node_sizes = [max(s, 100) for s in node_sizes]
    
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in main_G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_weights = [w * 2.0 / max_weight for w in edge_weights]
    
    num_nodes = len(main_G.nodes())
    if num_nodes > 50:
        figsize = (32, 24)
        font_size = 6
    elif num_nodes > 30:
        figsize = (24, 20)
        font_size = 8
    else:
        figsize = (18, 15)
        font_size = 10
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(main_G, k=2, iterations=50)
    
    nx.draw_networkx_nodes(main_G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(main_G, pos, width=edge_weights, 
                          alpha=0.5, edge_color='gray')
    
    labels = {node: node for node in main_G.nodes()}
    nx.draw_networkx_labels(main_G, pos, labels, font_size=font_size,
                           font_family='sans-serif',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor='none', alpha=0.7))
    
    plt.title(f"人物关系图 - {book_name or '未知'} (共{num_nodes}个人物)", 
              fontsize=14, pad=20)
    plt.axis('off')
    
    if save_path is None:
        save_path = "output"
    os.makedirs(save_path, exist_ok=True)
    
    if book_name:
        safe_book_name = re.sub(r'[<>:"/\\|?*]', '_', book_name)
        filename = os.path.join(save_path, f"{safe_book_name}_relationship.png")
    else:
        filename = os.path.join(save_path, "relationship.png")
    
    plt.savefig(filename, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ 已保存关系图: {filename}")
    plt.close()


def export_to_excel(relationships, entities, file_path, book_name=None):
    """导出人物关系到 Excel 文件"""
    rel_data = []
    for person1, relation, person2, confidence in relationships:
        rel_data.append({
            '人物1': person1,
            '关系': relation if relation else '相关',
            '人物2': person2,
            '置信度': f"{confidence:.4f}" if confidence > 0 else "N/A"
        })
    
    entity_data = []
    entity_counts = defaultdict(int)
    for person1, _, person2, _ in relationships:
        entity_counts[person1] += 1
        entity_counts[person2] += 1
    
    for entity in entities:
        entity_data.append({
            '人物': entity,
            '关系数量': entity_counts.get(entity, 0)
        })
    entity_data.sort(key=lambda x: x['关系数量'], reverse=True)
    
    relation_type_counts = defaultdict(int)
    for _, relation, _, _ in relationships:
        rel_type = relation if relation else '相关'
        relation_type_counts[rel_type] += 1
    
    rel_type_data = []
    for rel_type, count in sorted(relation_type_counts.items(), 
                                  key=lambda x: x[1], reverse=True):
        rel_type_data.append({
            '关系类型': rel_type,
            '出现次数': count
        })
    
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        if rel_data:
            df_rel = pd.DataFrame(rel_data)
            df_rel.to_excel(writer, sheet_name='关系详情', index=False)
        
        if entity_data:
            df_entity = pd.DataFrame(entity_data)
            df_entity.to_excel(writer, sheet_name='人物统计', index=False)
        
        if rel_type_data:
            df_rel_type = pd.DataFrame(rel_type_data)
            df_rel_type.to_excel(writer, sheet_name='关系类型统计', index=False)
    
    print(f"✅ 已导出 Excel 文件: {file_path}")


def sanitize_filename(filename):
    """清理文件名"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.strip(' .')
    if len(filename) > 100:
        filename = filename[:100]
    return filename


def main():
    parser = argparse.ArgumentParser(description="使用在线 API 提取小说人物关系（优化版）")
    parser.add_argument("--book", default="冬日重现", type=str,
                       help="书的名字，不带后缀")
    parser.add_argument("--base_url", type=str, 
                       default=os.getenv('API_BASE_URL', 'https://miaodi.zeabur.app'),
                       help="API 基础 URL（默认从环境变量 API_BASE_URL 读取）")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API 密钥（也可通过环境变量 API_KEY 设置）")
    parser.add_argument("--model", type=str,
                       default=os.getenv('API_MODEL', 'deepseek-ai/DeepSeek-V3-0324'),
                       help="模型名称（默认从环境变量 API_MODEL 读取）")
    parser.add_argument("--max_sentences", type=int, default=200,
                       help="最多提取的句子数（默认 200）")
    parser.add_argument("--context_lines", type=int, default=3,
                       help="段落上下文行数（默认 3）")
    parser.add_argument("--output", default="output", type=str,
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 文件路径
    fp = f"book/{args.book}.txt"
    if not os.path.exists(fp):
        print(f"❌ 错误: 文件不存在: {fp}")
        return
    
    print(f"=====+++=== 使用优化策略分析: {args.book} ===+++=====")
    print(f"📡 API 地址: {args.base_url}")
    print(f"📦 模型: {args.model}")
    
    # 提取关系
    try:
        relationships, entities = extract_relationships_optimized(
            fp, 
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model,
            max_sentences=args.max_sentences,
            context_lines=args.context_lines
        )
        
        if not relationships:
            print("⚠️ 未提取到任何关系")
            return
        
        # 构建关系图
        print("\n📊 正在构建关系图...")
        G, rel_dict = build_relationship_graph(relationships, entities)
        
        # 绘制关系图
        print("🎨 正在绘制关系图...")
        os.makedirs(args.output, exist_ok=True)
        plot_relationship_graph(G, relationships, 
                               save_path=args.output, 
                               book_name=args.book)
        
        # 导出 Excel
        print("📝 正在导出 Excel 文件...")
        excel_path = os.path.join(args.output, 
                                 f"{sanitize_filename(args.book)}_人物关系.xlsx")
        export_to_excel(relationships, entities, excel_path, book_name=args.book)
        
        print("\n" + "="*50)
        print("✅ 处理完成！")
        print(f"   - 提取到 {len(entities)} 个人物")
        print(f"   - 提取到 {len(relationships)} 个关系")
        print(f"   - 关系图已保存到: {args.output}")
        print(f"   - Excel 文件已保存到: {excel_path}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

