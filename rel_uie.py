# encoding=utf-8
# author： s0mE
# subject： 使用 PaddleNLP UIE 进行人物关系提取
# date： 2024
import argparse
import os
import re
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

try:
    from paddlenlp import Taskflow
    UIE_AVAILABLE = True
except ImportError:
    UIE_AVAILABLE = False
    print("⚠️ 警告: PaddleNLP 未安装，请运行: pip install paddlenlp")

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


def extract_relationships_with_uie(file_path, schema=None, batch_size=32, max_length=512):
    """
    使用 PaddleNLP UIE 从小说文本中提取人物关系
    
    Args:
        file_path: 小说文件路径
        schema: UIE 的抽取模式，默认为人物关系抽取
        batch_size: 批处理大小
        max_length: 最大文本长度
    
    Returns:
        relationships: 关系列表，格式为 [(person1, relation, person2, confidence), ...]
        entities: 实体列表
    """
    if not UIE_AVAILABLE:
        raise ImportError("PaddleNLP 未安装，请运行: pip install paddlenlp")
    
    # 默认 schema：抽取人物及其关系
    # UIE 支持多种 schema 格式，这里使用关系抽取模式
    if schema is None:
        # 方式1：关系抽取模式 - 直接定义关系三元组
        schema = [
            {'人物': ['关系']},  # 抽取 (人物, 关系) 对
            {'人物': ['人物']}   # 抽取 (人物, 人物) 对，用于共现关系
        ]
    
    # 初始化 UIE 模型
    # 支持通过环境变量选择模型（在 CI 环境中可以使用更小的模型）
    model_name = os.getenv('UIE_MODEL', 'uie-base')  # 默认使用 uie-base
    # 在 CI 环境中，如果没有 GPU，使用更小的模型
    if os.getenv('CI') == 'true':
        # uie-nano 是最小的模型，适合 CI 环境
        # uie-tiny 是较小的模型，适合资源受限环境
        # 如果环境变量未设置，在 CI 中使用 uie-tiny
        if model_name == 'uie-base':
            model_name = 'uie-tiny'  # CI 环境默认使用较小模型
            print("⚠️ CI 环境检测到，使用轻量级模型 uie-tiny（可通过 UIE_MODEL 环境变量覆盖）")
    
    print(f"📦 正在加载 PaddleNLP UIE 模型: {model_name}...")
    print("   💡 提示: 首次运行会下载模型文件（可能需要几分钟）")
    
    try:
        # 尝试使用指定的模型
        if model_name in ['uie-base', 'uie-medium', 'uie-mini', 'uie-micro', 'uie-nano', 'uie-tiny']:
            ie = Taskflow('information_extraction', 
                          schema=schema,
                          task_path=model_name,
                          batch_size=batch_size,
                          max_length=max_length)
        else:
            # 如果指定了自定义路径或其他模型名
            ie = Taskflow('information_extraction', 
                          schema=schema,
                          task_path=model_name,
                          batch_size=batch_size,
                          max_length=max_length)
    except Exception as e:
        print(f"⚠️ 模型 {model_name} 加载失败: {e}")
        print("   尝试使用默认模型配置...")
        # 如果指定模型路径失败，使用默认模型
        try:
            ie = Taskflow('information_extraction', 
                          schema=schema,
                          batch_size=batch_size,
                          max_length=max_length)
        except Exception as e2:
            print(f"❌ 默认模型加载也失败: {e2}")
            raise
    
    # 读取文本文件
    print("📖 正在读取文本文件...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk") as f:
            lines = f.readlines()
    
    # 过滤空行和过短的行
    lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    relationships = []
    entities = set()
    all_results = []
    
    print("🔍 正在使用 UIE 提取人物关系...")
    # 分批处理文本
    batch_texts = []
    for i, line in enumerate(tqdm(lines, desc="Processing")):
        # 清理文本
        line = re.sub(r'\s+', '', line)
        if len(line) < 10:
            continue
        
        batch_texts.append(line)
        
        # 当达到批处理大小时，进行抽取
        if len(batch_texts) >= batch_size:
            try:
                results = ie(batch_texts)
                all_results.extend(results)
                batch_texts = []
            except Exception as e:
                print(f"⚠️ 处理批次时出错: {e}")
                batch_texts = []
                continue
    
    # 处理剩余的文本
    if batch_texts:
        try:
            results = ie(batch_texts)
            all_results.extend(results)
        except Exception as e:
            print(f"⚠️ 处理最后批次时出错: {e}")
    
    # 解析结果
    print("📊 正在解析抽取结果...")
    for result in tqdm(all_results, desc="Parsing"):
        if not result or not isinstance(result, dict):
            continue
        
        # UIE 的结果格式可能是多种形式，需要灵活处理
        # 方式1: 如果 schema 是 {'人物': ['关系']}，结果是 {人物: [{text: ..., 关系: [{text: ...}]}]}
        if '人物' in result:
            persons = result['人物']
            if isinstance(persons, list):
                for person_info in persons:
                    if isinstance(person_info, dict):
                        person_name = person_info.get('text', '')
                        if not person_name:
                            continue
                        
                        entities.add(person_name)
                        
                        # 提取关系
                        relations = person_info.get('关系', [])
                        if isinstance(relations, list):
                            for rel_info in relations:
                                if isinstance(rel_info, dict):
                                    rel_type = rel_info.get('text', '')
                                    # 如果关系指向另一个实体
                                    if '人物' in rel_info:
                                        related_persons = rel_info['人物']
                                        if not isinstance(related_persons, list):
                                            related_persons = [related_persons]
                                        for related_person_info in related_persons:
                                            if isinstance(related_person_info, dict):
                                                related_person = related_person_info.get('text', '')
                                            else:
                                                related_person = str(related_person_info)
                                            
                                            if related_person and related_person != person_name:
                                                entities.add(related_person)
                                                confidence = rel_info.get('probability', 
                                                                         related_person_info.get('probability', 0.0) if isinstance(related_person_info, dict) else 0.0)
                                                relationships.append((
                                                    person_name,
                                                    rel_type if rel_type else '相关',
                                                    related_person,
                                                    confidence
                                                ))
        
        # 方式2: 如果 schema 是 {'人物': ['人物']}，结果是共现关系
        # 这种方式提取的是在同一句话中出现的两个人
        # 注意：这种方式需要额外的文本上下文，UIE 可能不会直接返回这种格式
        
        # 方式3: 处理关系抽取的另一种格式 - 直接的三元组形式
        # 如果结果包含 'relation' 字段
        if 'relation' in result:
            for rel_entry in result['relation'] if isinstance(result['relation'], list) else [result['relation']]:
                if isinstance(rel_entry, dict):
                    subject = rel_entry.get('subject', {}).get('text', '') if isinstance(rel_entry.get('subject'), dict) else ''
                    object_entity = rel_entry.get('object', {}).get('text', '') if isinstance(rel_entry.get('object'), dict) else ''
                    predicate = rel_entry.get('predicate', '')
                    if subject and object_entity:
                        entities.add(subject)
                        entities.add(object_entity)
                        relationships.append((
                            subject,
                            predicate if predicate else '相关',
                            object_entity,
                            rel_entry.get('probability', 0.0)
                        ))
    
    print(f"✅ 提取完成: 发现 {len(entities)} 个人物，{len(relationships)} 个关系")
    return relationships, list(entities)


def build_relationship_graph(relationships, entities=None):
    """
    构建人物关系图
    
    Args:
        relationships: 关系列表
        entities: 实体列表（可选）
    
    Returns:
        G: NetworkX 图对象
        rel_dict: 关系字典 {(person1, person2): [relations]}
    """
    G = nx.Graph()
    rel_dict = defaultdict(list)
    
    # 添加节点
    if entities:
        for entity in entities:
            G.add_node(entity)
    
    # 添加边和关系
    for person1, relation, person2, confidence in relationships:
        if person1 and person2 and person1 != person2:
            G.add_node(person1)
            G.add_node(person2)
            G.add_edge(person1, person2, weight=1.0, relation=relation, confidence=confidence)
            rel_dict[(person1, person2)].append((relation, confidence))
    
    return G, dict(rel_dict)


def plot_relationship_graph(G, relationships, save_path=None, book_name=None):
    """
    绘制人物关系图
    
    Args:
        G: NetworkX 图对象
        relationships: 关系列表
        save_path: 保存路径
        book_name: 书籍名称
    """
    if not G.nodes():
        print("⚠️ 图中没有节点，无法绘制")
        return
    
    # 计算节点度（连接数）
    degrees = dict(G.degree())
    
    # 选择主要子图
    if nx.is_connected(G):
        main_G = G
    else:
        components = list(nx.connected_components(G))
        main_component = max(components, key=len)
        main_G = G.subgraph(main_component).copy()
        print(f"📊 主要子图包含 {len(main_component)} 个节点（共 {len(G.nodes())} 个节点）")
    
    # 计算节点大小
    node_sizes = [degrees.get(node, 1) * 500 for node in main_G.nodes()]
    node_sizes = [max(s, 100) for s in node_sizes]  # 最小尺寸
    
    # 计算边的权重
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in main_G.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        edge_weights = [w * 2.0 / max_weight for w in edge_weights]
    
    # 根据节点数量调整画布大小
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
    
    # 绘制图形
    plt.figure(figsize=figsize)
    
    # 使用 spring 布局
    pos = nx.spring_layout(main_G, k=2, iterations=50)
    
    # 绘制节点和边
    nx.draw_networkx_nodes(main_G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, 
                          edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(main_G, pos, width=edge_weights, 
                          alpha=0.5, edge_color='gray')
    
    # 绘制标签
    labels = {node: node for node in main_G.nodes()}
    nx.draw_networkx_labels(main_G, pos, labels, font_size=font_size,
                           font_family='sans-serif',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor='none', alpha=0.7))
    
    plt.title(f"人物关系图 - {book_name or '未知'} (共{num_nodes}个人物)", 
              fontsize=14, pad=20)
    plt.axis('off')
    
    # 保存图片
    if save_path is None:
        save_path = "output"
    os.makedirs(save_path, exist_ok=True)
    
    if book_name:
        safe_book_name = re.sub(r'[<>:"/\\|?*]', '_', book_name)
        filename = os.path.join(save_path, f"{safe_book_name}_uie_relationship.png")
    else:
        filename = os.path.join(save_path, "uie_relationship.png")
    
    plt.savefig(filename, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ 已保存关系图: {filename}")
    plt.close()


def export_to_excel(relationships, entities, file_path, book_name=None):
    """
    导出人物关系到 Excel 文件
    
    Args:
        relationships: 关系列表
        entities: 实体列表
        file_path: 输出文件路径
        book_name: 书籍名称
    """
    # 创建多个工作表的数据
    
    # 1. 关系详情表
    rel_data = []
    for person1, relation, person2, confidence in relationships:
        rel_data.append({
            '人物1': person1,
            '关系': relation if relation else '相关',
            '人物2': person2,
            '置信度': f"{confidence:.4f}" if confidence > 0 else "N/A"
        })
    
    # 2. 人物统计表
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
    
    # 3. 关系类型统计表
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
    
    # 写入 Excel
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # 关系详情
        if rel_data:
            df_rel = pd.DataFrame(rel_data)
            df_rel.to_excel(writer, sheet_name='关系详情', index=False)
        
        # 人物统计
        if entity_data:
            df_entity = pd.DataFrame(entity_data)
            df_entity.to_excel(writer, sheet_name='人物统计', index=False)
        
        # 关系类型统计
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
    parser = argparse.ArgumentParser(description="使用 PaddleNLP UIE 提取小说人物关系")
    parser.add_argument("--book", default="冬日重现", type=str,
                       help="书的名字，不带后缀")
    parser.add_argument("--schema", type=str, default=None,
                       help="自定义 schema（JSON 格式），默认使用人物关系抽取")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批处理大小")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大文本长度")
    parser.add_argument("--output", default="output", type=str,
                       help="输出目录")
    
    args = parser.parse_args()
    
    if not UIE_AVAILABLE:
        print("❌ 错误: PaddleNLP 未安装")
        print("请运行: pip install paddlenlp")
        return
    
    # 文件路径
    fp = f"book/{args.book}.txt"
    if not os.path.exists(fp):
        print(f"❌ 错误: 文件不存在: {fp}")
        return
    
    print(f"=====+++=== 使用 PaddleNLP UIE 分析: {args.book} ===+++=====")
    
    # 解析 schema（如果提供）
    schema = None
    if args.schema:
        import json
        try:
            schema = json.loads(args.schema)
        except:
            print("⚠️ Schema 解析失败，使用默认 schema")
    
    # 提取关系
    try:
        relationships, entities = extract_relationships_with_uie(
            fp, 
            schema=schema,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        if not relationships:
            print("⚠️ 未提取到任何关系，可能需要调整 schema 或文本格式")
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
                                 f"{sanitize_filename(args.book)}_人物关系.xlsx")
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

