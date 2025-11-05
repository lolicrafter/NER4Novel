# encoding=utf-8
# author： s0mE
# subject： 使用在线 API 进行人物关系提取（支持 OpenAI、智谱、DeepSeek 等）
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


class LLMAPI:
    """统一的 LLM API 接口"""
    
    def __init__(self, provider='openai', api_key=None, base_url=None):
        """
        Args:
            provider: API 提供商 ('openai', 'zhipu', 'deepseek', 'moonshot', 'qwen')
            api_key: API 密钥
            base_url: API 基础 URL（可选，用于自定义端点）
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv(f'{provider.upper()}_API_KEY') or os.getenv('API_KEY')
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(f"需要设置 API 密钥，通过参数或环境变量 {provider.upper()}_API_KEY")
        
        # 根据提供商设置默认 base_url
        if not self.base_url:
            if self.provider == 'openai':
                self.base_url = 'https://api.openai.com/v1'
            elif self.provider == 'zhipu':
                self.base_url = 'https://open.bigmodel.cn/api/paas/v4'
            elif self.provider == 'deepseek':
                self.base_url = 'https://api.deepseek.com/v1'
            elif self.provider == 'moonshot':
                self.base_url = 'https://api.moonshot.cn/v1'
            elif self.provider == 'qwen':
                self.base_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 设置模型名称
        self.model_map = {
            'openai': 'gpt-3.5-turbo',
            'zhipu': 'glm-4',
            'deepseek': 'deepseek-chat',
            'moonshot': 'moonshot-v1-8k',
            'qwen': 'qwen-turbo'
        }
        self.model = self.model_map.get(self.provider, 'gpt-3.5-turbo')
    
    def call_api(self, prompt, max_tokens=2000, temperature=0.3):
        """调用 API"""
        import requests
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        if self.provider == 'openai':
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = f'{self.base_url}/chat/completions'
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        elif self.provider == 'zhipu':
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = f'{self.base_url}/chat/completions'
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        elif self.provider == 'deepseek':
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = f'{self.base_url}/chat/completions'
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        elif self.provider == 'moonshot':
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = f'{self.base_url}/chat/completions'
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        elif self.provider == 'qwen':
            headers['Authorization'] = f'Bearer {self.api_key}'
            url = f'{self.base_url}/services/aigc/text-generation/generation'
            data = {
                'model': self.model,
                'input': {'messages': [{'role': 'user', 'content': prompt}]},
                'parameters': {
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            }
        else:
            raise ValueError(f"不支持的提供商: {self.provider}")
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            if self.provider == 'qwen':
                return result.get('output', {}).get('text', '')
            else:
                return result.get('choices', [{}])[0].get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            print(f"❌ API 调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   响应内容: {e.response.text}")
            raise


def extract_relationships_with_llm(file_path, api_provider='openai', api_key=None, 
                                   batch_size=10, max_length=500):
    """
    使用大语言模型 API 从小说文本中提取人物关系
    
    Args:
        file_path: 小说文件路径
        api_provider: API 提供商
        api_key: API 密钥
        batch_size: 批处理大小（每批处理的句子数）
        max_length: 最大文本长度
    
    Returns:
        relationships: 关系列表，格式为 [(person1, relation, person2, confidence), ...]
        entities: 实体列表
    """
    # 初始化 API
    try:
        llm = LLMAPI(provider=api_provider, api_key=api_key)
        print(f"✅ 已连接到 {api_provider.upper()} API")
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
    
    # 过滤和清理文本
    lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    # 构建提示词模板
    prompt_template = """你是一个专业的小说分析助手。请从以下文本中提取人物关系。

要求：
1. 识别文本中出现的所有人物姓名
2. 提取人物之间的关系（如：父子、朋友、恋人、同事、敌人、师生等）
3. 如果关系不明确，使用"相关"作为关系类型

输出格式为 JSON 数组，每个元素格式如下：
{
  "person1": "人物1",
  "relation": "关系类型",
  "person2": "人物2"
}

文本内容：
{text}

请只返回 JSON 数组，不要包含其他解释文字。如果文本中没有人物关系，返回空数组 []。"""

    relationships = []
    entities = set()
    
    print("🔍 正在使用 LLM API 提取人物关系...")
    
    # 分批处理文本
    batch_texts = []
    current_batch = ""
    
    for i, line in enumerate(tqdm(lines, desc="Processing")):
        # 清理文本
        line = re.sub(r'\s+', '', line)
        if len(line) < 10:
            continue
        
        # 累积文本
        if len(current_batch) + len(line) < max_length:
            current_batch += line + "。"
        else:
            if current_batch:
                batch_texts.append(current_batch)
                current_batch = line + "。"
            else:
                # 如果单行太长，截断
                current_batch = line[:max_length] + "。"
        
        # 当达到批处理大小时，进行抽取
        if len(batch_texts) >= batch_size:
            try:
                # 合并批次文本
                combined_text = "\n".join(batch_texts)
                prompt = prompt_template.format(text=combined_text[:2000])  # 限制总长度
                
                # 调用 API
                response = llm.call_api(prompt, max_tokens=2000, temperature=0.3)
                
                # 解析响应
                try:
                    # 尝试提取 JSON
                    json_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if json_match:
                        rel_data = json.loads(json_match.group())
                    else:
                        # 如果没有找到 JSON，尝试直接解析
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
                    print(f"⚠️ JSON 解析失败: {e}")
                    print(f"   响应内容: {response[:200]}")
                
                batch_texts = []
                current_batch = ""
                
                # 避免 API 限流
                time.sleep(0.5)
                
            except Exception as e:
                print(f"⚠️ 处理批次时出错: {e}")
                batch_texts = []
                current_batch = ""
                continue
    
    # 处理剩余文本
    if current_batch:
        batch_texts.append(current_batch)
    
    if batch_texts:
        try:
            combined_text = "\n".join(batch_texts)
            prompt = prompt_template.format(text=combined_text[:2000])
            response = llm.call_api(prompt, max_tokens=2000, temperature=0.3)
            
            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    rel_data = json.loads(json_match.group())
                else:
                    rel_data = json.loads(response)
                
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
                                0.8
                            ))
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON 解析失败: {e}")
        except Exception as e:
            print(f"⚠️ 处理最后批次时出错: {e}")
    
    print(f"✅ 提取完成: 发现 {len(entities)} 个人物，{len(relationships)} 个关系")
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
        filename = os.path.join(save_path, f"{safe_book_name}_api_relationship.png")
    else:
        filename = os.path.join(save_path, "api_relationship.png")
    
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
    parser = argparse.ArgumentParser(description="使用在线 API 提取小说人物关系")
    parser.add_argument("--book", default="冬日重现", type=str,
                       help="书的名字，不带后缀")
    parser.add_argument("--provider", default="openai", type=str,
                       choices=['openai', 'zhipu', 'deepseek', 'moonshot', 'qwen'],
                       help="API 提供商")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API 密钥（也可通过环境变量设置）")
    parser.add_argument("--base_url", type=str, default=None,
                       help="自定义 API 基础 URL")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="批处理大小（每批处理的句子数）")
    parser.add_argument("--max_length", type=int, default=500,
                       help="最大文本长度")
    parser.add_argument("--output", default="output", type=str,
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 文件路径
    fp = f"book/{args.book}.txt"
    if not os.path.exists(fp):
        print(f"❌ 错误: 文件不存在: {fp}")
        return
    
    print(f"=====+++=== 使用 {args.provider.upper()} API 分析: {args.book} ===+++=====")
    
    # 提取关系
    try:
        relationships, entities = extract_relationships_with_llm(
            fp, 
            api_provider=args.provider,
            api_key=args.api_key,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        if not relationships:
            print("⚠️ 未提取到任何关系，可能需要调整参数或检查 API 配置")
            return
        
        # 构建关系图
        print("📊 正在构建关系图...")
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
                                 f"{sanitize_filename(args.book)}_人物关系_api.xlsx")
        export_to_excel(relationships, entities, excel_path, book_name=args.book)
        
        print("="*50)
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

