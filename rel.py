# encoding=utf-8
# author： s0mE
# subject： 人名以及关系提取
# date： 2019-06-26
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
from pyhanlp import *

# 尝试导入 OpenAI 库（用于调用 DeepSeek API）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ openai 库未安装，LLM 分析功能不可用。安装方法: pip install openai")

# 尝试导入 pandas（用于 Excel 导出）
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas 库未安装，Excel 导出功能不可用。安装方法: pip install pandas openpyxl")



# 设置中文字体 - 尝试多个字体选项以支持不同环境
# 优先使用 SimHei（Windows/本地），如果不可用则使用其他支持中文的字体
import matplotlib.font_manager as fm

# 获取系统可用的中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                 'Noto Sans CJK SC', 'Source Han Sans CN', 'Droid Sans Fallback', 'DejaVu Sans']
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 找到第一个可用的中文字体
font_found = None
for font in chinese_fonts:
    if font in available_fonts:
        font_found = font
        break

if font_found:
    plt.rcParams["font.sans-serif"] = [font_found] + plt.rcParams["font.sans-serif"]
else:
    # 如果没有找到中文字体，使用默认字体并尝试设置
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] + plt.rcParams["font.sans-serif"]
    print("⚠️ Warning: No Chinese font found, Chinese characters may display as squares")

plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class hanlp(object):
    def __init__(self, analyzer = "Perceptron", custom_dict = True ):
        ## 数据集目录 - 动态获取 pyhanlp 安装路径
        import pyhanlp
        
        # 获取 pyhanlp 的安装路径
        pyhanlp_dir = os.path.dirname(pyhanlp.__file__)
        static_dir = os.path.join(pyhanlp_dir, 'static')
        data_path = os.path.join(static_dir, 'data', 'model', 'perceptron', 'large', 'cws.bin')
        
        # 如果文件不存在，使用 HanLP 的默认配置（让 HanLP 自动查找数据文件）
        if not os.path.exists(data_path):
            data_path = None
        
        ## 构造人名分析器
        # 常规识别
        # self.analyzer = HanLP.newSegment().enableNameRecognize(True)

        # # crf识别
        self.CRFLAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")()

        #感知机识别
        _PLAnalyzer = JClass("com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer")
        if data_path:
            # 使用指定的路径
            self.PLAnalyzer = _PLAnalyzer(
                data_path, HanLP.Config.PerceptronPOSModelPath, HanLP.Config.PerceptronNERModelPath)
        else:
            # 使用默认配置（HanLP 会自动查找数据文件）
            self.PLAnalyzer = _PLAnalyzer()
        
        self.analyzer = self.PLAnalyzer
        if analyzer=="Perceptron":
            self.analyzer = self.PLAnalyzer.enableCustomDictionary(custom_dict)
        elif analyzer=="CRF":
            self.analyzer = self.CRFLAnalyzer.enableCustomDictionary(custom_dict)
        
        # Cache JString class for type conversion
        self.JString = JClass("java.lang.String")
        
    def cut(self, words):
        res = []
        # Convert Python string to Java String for JPype1 compatibility
        # This resolves ambiguous overload between seg(String) and seg(char[])
        if isinstance(words, str):
            words = self.JString(words)
        
        if self.analyzer is None:
            terms = HanLP.segment(words)
        else:
            # Use explicit method call to avoid overload ambiguity
            terms = self.analyzer.seg(words)
        for term in terms:
            res.append( (str(term.word),str(term.nature)) )
        return res
    
    @classmethod
    def add(self,names_list):
        for n in names_list:
            if CustomDictionary.get(n) is None:
                CustomDictionary.add(n,"nr 1000 ")
            else:
                attr = "nr 1000 " + str(CustomDictionary.get(n))
                # attr = "nr 1000 "
                CustomDictionary.insert(n,attr)

    @classmethod
    def insert(self, names_list):
        for n in names_list:
            CustomDictionary.insert(n, "nr 1")
            
def count_names(fp,model):
    """
    统计文本中的所有名字，返回统计矩阵
    """
    #逐行提取名字
    name_set = set() # 所有名字的集合
    
    
    nr_nrf_dict = {"nr":{},"nrf":{}}

    cut_result = []
    lines = []

    try:
        with open(fp, "r") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(fp,"r",encoding="gbk") as f:
            lines = f.readlines()

    for line in tqdm(lines, desc="Analyzing"):
        #每一行做预处理
        line = line.strip().replace(" ","")

        words = model.cut(line)
        line_dict = {}

        for word, flag in words:
            # if word == "张":
            #     print(word,flag,"|||",line)
            
            if flag == "nr" or flag == "nrf":# or flag == "j":
                # 如果 word 是人名，加入人名的统计中
                line_dict[word] = line_dict.get(word, 0) + 1
                name_set.add(word)

                # 分中文名和英文名统计名称
                nr_nrf_dict[flag][word] = nr_nrf_dict[flag].get(word, 0) + 1
                
        if len(line_dict) != 0:
            cut_result.append(line_dict)

    # 名字关系矩阵计算
    names = list(name_set)  # 所有名字的列表
    name_arr = np.zeros((len(names), len(cut_result)),
                        dtype=np.int32)  # 储存统计结果的数组
    for n, n_dict in enumerate(cut_result):
            for k, v in n_dict.items():
                i = names.index(k)
                name_arr[i, n] += v
    # 计算人名的关系矩阵
    names = np.array(names)
    rel = np.zeros((len(names), len(names)), dtype=np.int32)
    for i in range(len(names)):
        rel[i, :] = np.sum(name_arr[:, name_arr[i, :] > 0], axis=1)

    ########至此，已经初步完成了文章的人物关系统计##############
    ############ 不过这里仍然有很多问题   ###################
    #### 例如明显的错误名字，以及同一人物不同的别称需要进一步处理 ###
    ################需要后续的处理 #######################
    return rel, names, nr_nrf_dict


def filter_nr(nr_nrf_dict, threshold = -1,first=False):
    """
    自动生成可信名称列表 和 名字转换字典
    """
    nr_dict = nr_nrf_dict["nr"]
    nrf_dict = nr_nrf_dict["nrf"]
    
    first_threshold = 5
    if threshold == -1:
        threshold = np.mean( list(nr_dict.values())+list(nrf_dict.values()))
        first_threshold = max(np.sqrt(len(nr_dict)+len(nrf_dict)),5*threshold)
    print("auto_dict threshold:{:.3f}".format(threshold))
    names = []
    trans_dict = {}
    last_names = []
    last_repeat = []

    first_names = []
    first_repeat = []
    for name,value in sorted(nr_dict.items(), key=lambda d: d[1], reverse=True):
        if value > threshold:
            if len(name) == 1 and value < first_threshold:
                continue
            names.append(name)
            last_name = name[1:]
            # 获取三字姓名的名字的部分，如果存在重复的删除
            if len(name)==3 and not last_name in last_repeat:
                if last_name in last_names:
                    last_names.remove(last_name)
                    trans_dict.pop(last_name)
                    last_repeat.append(last_name)
                else:
                    trans_dict[last_name] = name
                    last_names.append(last_name)
            
            # 获取姓名的姓的部分
            first_name = name[:1]
            if first and len(name)==3 and not first_name in first_repeat:
                if first_name in first_names:
                    first_names.remove(first_name)
                    trans_dict.pop(first_name)
                    first_repeat.append(first_name)
                else:
                    trans_dict[first_name] = name
                    first_names.append(first_name)
        
    names = last_names + names
    # print(names)
    for name,value in nrf_dict.items():
        if value > threshold:
            names.append(name)
    return names,trans_dict

def filter_names(rel, names, trans={}, err=[], threshold= -1):
    """对结果进行精细的调整与过滤

    处理顺序: 转换 ==> 去错 ==> 去重（子串合并）==> 过滤 ==> 排序

    Args:
        rel:关系矩阵 n x n
        names: 人名向量矩阵 n
        trans: 别称转换字典 将别称转换为统一名字
        err: 错误名称矩阵 要删除的错误名称列表
        threshold: 词频阈值 词频低于此阈值的名字会被过滤，等于-1（default）时使用词频均值自动过滤
    
    Returns:
        rel_filter
        names_filter
        过滤好的人名矩阵和名称矩阵
    """
    
    rel = np.copy(rel)
    names = np.copy(names)

    # 名字的转换与计数的合并
    if len(trans) != 0:
        name_new = list(set(names) - set(trans.keys()))  # 转换后的名字
        indexes = [list(names).index(n) for n in name_new]
        for i, name in enumerate(names):
            if name in trans.keys():
                new_i = list(names).index(trans[names[i]])
                rel[new_i, :] += rel[i, :]
                rel[:, new_i] += rel[:, i]
        names = np.array(name_new)
        rel = rel[indexes, :][:, indexes]

    # 去错
    # 自动过滤掉明显不是人名的字符（如省略号、标点符号等）
    import re
    # 检查名字是否只包含标点符号、空白字符或特殊符号（不包含汉字、字母、数字）
    def is_invalid_name(name):
        # 如果名字为空或只包含空白字符，直接返回 True
        if not name or name.strip() == '':
            return True
        # 检查是否包含有效字符（汉字、字母、数字）
        if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', name):
            return False
        # 如果不包含任何有效字符，则认为是无效名字（只包含标点符号等）
        return True
    
    auto_err_list = []
    for name in names:
        if is_invalid_name(name):
            auto_err_list.append(name)
    
    # 合并手动错误列表和自动检测的错误列表
    all_err_list = list(set(err + auto_err_list))
    
    if len(all_err_list) != 0:
        name_new = list(set(names)-set(all_err_list))  # 去错后的名字列表
        indexes = [list(names).index(n) for n in name_new]
        names = np.array(name_new)
        rel = rel[indexes, :][:, indexes]
        if len(auto_err_list) > 0:
            print(f"✅ 自动过滤：删除了 {len(auto_err_list)} 个无效人名（标点符号等）: {sorted(auto_err_list)}")

    # 去重：如果一个较短的人名是另一个更长人名的子串，删除较短的人名
    # 例如："路青怜" 和 "路青" -> 保留 "路青怜"，删除 "路青"
    # 例如："顾秋绵" 和 "顾秋" -> 保留 "顾秋绵"，删除 "顾秋"
    names_list = names.tolist()
    names_to_remove = set()
    name_frequencies = {name: rel[names_list.index(name), names_list.index(name)] for name in names_list}
    
    # 按长度排序，先处理较短的名称
    sorted_names = sorted(names_list, key=lambda x: (len(x), -name_frequencies.get(x, 0)))
    
    for i, short_name in enumerate(sorted_names):
        if short_name in names_to_remove:
            continue
        
        short_freq = name_frequencies.get(short_name, 0)
        
        # 检查是否有更长的名字包含这个短名字
        for long_name in names_list:
            if long_name == short_name or long_name in names_to_remove:
                continue
            
            # 如果短名字是长名字的子串（前缀、后缀或中间部分）
            if short_name in long_name and len(short_name) < len(long_name):
                long_freq = name_frequencies.get(long_name, 0)
                
                # 如果长名字频率更高或相当（至少是短名字的 0.5 倍），则删除短名字
                # 这样可以处理 "路青怜" 和 "路青" 的情况
                if long_freq >= short_freq * 0.5:
                    names_to_remove.add(short_name)
                    # 将短名字的关系合并到长名字中
                    short_idx = names_list.index(short_name)
                    long_idx = names_list.index(long_name)
                    rel[long_idx, :] += rel[short_idx, :]
                    rel[:, long_idx] += rel[:, short_idx]
                    break
    
    # 移除需要删除的名字
    if names_to_remove:
        name_new = [n for n in names_list if n not in names_to_remove]
        indexes = [names_list.index(n) for n in name_new]
        names = np.array(name_new)
        rel = rel[indexes, :][:, indexes]
        print(f"✅ 去重处理：删除了 {len(names_to_remove)} 个重复子串人名: {sorted(names_to_remove)}")

    # 过滤掉低频的名字
    if threshold != 0:
        if threshold == -1:
            rel_threshold = max(rel.diagonal().mean(), threshold)
        else:
            rel_threshold = threshold
        print("out threshold:{:.3f}".format(rel_threshold))
        rel_filter = np.diag(rel) > rel_threshold
        names = names[rel_filter]
        rel = rel[rel_filter, :][:, rel_filter]
    

    # 人名排序
    indexes = np.argsort(np.diag(rel))[::-1]  # 从大到小
    names = names[indexes]
    rel = rel[indexes, :][:, indexes]

    # 限制最多显示60个人名（按出现频率从高到低）
    MAX_NAMES = 60
    original_count = len(names)
    if len(names) > MAX_NAMES:
        names = names[:MAX_NAMES]
        rel = rel[:MAX_NAMES, :][:, :MAX_NAMES]
        print(f"⚠️ 限制显示人数：保留前 {MAX_NAMES} 个高频人物（共 {original_count} 个）")

    # 打印所有人名
    print(f"所有人名: {names}")
    return rel, names


def find_paragraphs_with_two_names(text_lines, names_list, context_lines=3, max_paragraphs_per_person=20):
    """
    找出所有至少包含两个名字的段落，并根据人名限制段落数量
    
    Args:
        text_lines: 文本行列表
        names_list: 人名列表（应该是最终过滤后的人名，避免子串重复）
        context_lines: 段落上下文行数（前后各多少行）
        max_paragraphs_per_person: 每个人名最多保留的段落数
    
    Returns:
        paragraphs_data: [(paragraph, line_idx, found_names_list), ...]
    """
    # 过滤掉子串人名：如果短名字是长名字的子串，且在同一人名列表中，只保留长名字
    def filter_substring_names(names):
        """过滤掉是其他名字子串的名字"""
        names_unique = list(set(names))
        names_sorted = sorted(names_unique, key=len, reverse=True)
        filtered = []
        
        for name in names_sorted:
            # 检查这个名字是否是已保留名字的子串
            is_substring = False
            for kept_name in filtered:
                if name in kept_name and name != kept_name:
                    is_substring = True
                    break
            if not is_substring:
                filtered.append(name)
        
        return filtered
    
    # 过滤子串人名
    names_filtered = filter_substring_names(names_list)
    print(f"📋 过滤子串人名: {len(names_list)} -> {len(names_filtered)} 个")
    if len(names_list) != len(names_filtered):
        removed = set(names_list) - set(names_filtered)
        print(f"   移除的子串人名: {sorted(removed)}")
    
    # 构建人名匹配模式（按长度排序，优先匹配长名字）
    names_sorted = sorted(names_filtered, key=len, reverse=True)
    
    # 第一遍：找出所有包含至少两个人名的段落
    all_paragraphs = []
    
    # 使用更精确的去重方式：存储段落内容本身，而不是hash（hash可能冲突）
    seen_paragraph_texts = set()
    
    # 为了进一步去重，记录每个段落的唯一标识（基于内容和行号范围）
    seen_paragraph_keys = set()
    
    for line_idx in range(len(text_lines)):
        # 提取段落上下文
        paragraph = extract_paragraph_context(text_lines, line_idx, context_lines)
        
        # 去重方式1：使用段落内容本身作为键（避免hash冲突）
        if paragraph in seen_paragraph_texts:
            continue
        
        # 去重方式2：使用段落内容+行号范围作为唯一键（避免相邻行产生的重复段落）
        paragraph_key = (paragraph, line_idx // (context_lines * 2 + 1))  # 按段落区域分组
        if paragraph_key in seen_paragraph_keys:
            continue
        
        seen_paragraph_texts.add(paragraph)
        seen_paragraph_keys.add(paragraph_key)
        
        # 找出段落中出现的所有人名（使用精确匹配，避免子串误匹配）
        found_names = []
        for name in names_sorted:
            # 使用更精确的匹配：确保是完整词匹配，而不是子串匹配
            # 检查 name 是否作为独立词出现在段落中
            if name in paragraph:
                # 进一步检查：确保不是其他名字的一部分（已通过排序避免）
                found_names.append(name)
        
        # 如果找到至少两个人名，记录下来
        if len(found_names) >= 2:
            all_paragraphs.append((paragraph, line_idx, found_names))
    
    print(f"✅ 找到 {len(all_paragraphs)} 个包含至少两个人名的段落")
    
    # 第二遍：按人名限制段落数量，每个人最多保留 max_paragraphs_per_person 个段落
    person_paragraph_count = defaultdict(int)  # 统计每个人名已经保留的段落数
    selected_paragraphs = []
    
    # 按行号排序，保持顺序
    all_paragraphs.sort(key=lambda x: x[1])
    
    for paragraph, line_idx, found_names in all_paragraphs:
        # 检查这个段落中是否还有未达到上限的人名
        can_add = False
        for name in found_names:
            if person_paragraph_count[name] < max_paragraphs_per_person:
                can_add = True
                break
        
        if can_add:
            # 添加这个段落，并更新计数
            selected_paragraphs.append((paragraph, line_idx, found_names))
            for name in found_names:
                person_paragraph_count[name] += 1
    
    print(f"✅ 限制后保留 {len(selected_paragraphs)} 个段落（每个人最多 {max_paragraphs_per_person} 个）")
    
    # 打印统计信息
    print(f"\n📊 人名段落统计（前10个）:")
    sorted_persons = sorted(person_paragraph_count.items(), key=lambda x: x[1], reverse=True)
    for name, count in sorted_persons[:10]:
        print(f"   {name}: {count} 个段落")
    
    return selected_paragraphs


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


def analyze_relationships_with_llm(text_lines, names_list, base_url, api_key, model_name,
                                   max_sentences=200, context_lines=3):
    """
    使用 LLM（DeepSeek）分析人物关系
    
    Args:
        text_lines: 文本行列表
        names_list: 人名列表
        base_url: API 基础 URL（如 https://api.deepseek.com）
        api_key: API 密钥
        model_name: 模型名称（如 deepseek-reasoner 或 deepseek-chat）
        max_sentences: 最多分析的句子数
        context_lines: 段落上下文行数
    
    Returns:
        relationships: [(person1, relation, person2, weight), ...]
        all_names: 所有人名集合
        paragraphs_data: [(paragraph, line_idx, person1, person2, sentence), ...] 段落数据列表
    """
    if not OPENAI_AVAILABLE:
        print("❌ OpenAI 库未安装，无法使用 LLM 分析")
        return [], set(names_list), []
    
    # 初始化 OpenAI 客户端（DeepSeek 兼容 OpenAI API）
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip('/')
        )
        print(f"✅ 已连接到 DeepSeek API: {base_url}")
        print(f"📦 使用模型: {model_name}")
    except Exception as e:
        print(f"❌ API 初始化失败: {e}")
        return [], set(names_list), []
    
    # 阶段1: 找出所有包含至少两个人名的段落
    print(f"\n🔍 阶段1: 找出所有包含至少两个人名的段落...")
    paragraphs_with_names = find_paragraphs_with_two_names(
        text_lines, names_list, context_lines=context_lines, max_paragraphs_per_person=20
    )
    
    if len(paragraphs_with_names) == 0:
        print("⚠️ 未找到包含至少两个人名的段落")
        return [], set(names_list), []
    
    # 准备段落数据用于导出 Excel（进一步去重）
    paragraphs_data_for_excel = []
    unique_paragraphs = []
    
    # 用于记录已导出的段落，避免重复
    exported_paragraphs = set()
    
    for paragraph, line_idx, found_names in paragraphs_with_names:
        unique_paragraphs.append((paragraph, line_idx))
        
        # 为 Excel 导出准备数据：列出所有可能的人名对
        # 但每个段落只导出一次（基于段落内容）
        paragraph_text = paragraph.strip()
        if paragraph_text in exported_paragraphs:
            continue  # 跳过重复段落
        exported_paragraphs.add(paragraph_text)
        
        # 为每个人名对创建一条记录
        for i in range(len(found_names)):
            for j in range(i + 1, len(found_names)):
                person1, person2 = found_names[i], found_names[j]
                if person1 != person2:
                    # 提取段落中的句子（用于显示）
                    sentence_pattern = r'[。！？；\n]+'
                    sentences = re.split(sentence_pattern, paragraph)
                    # 找到包含这两个人名的句子
                    relevant_sentence = ""
                    for sent in sentences:
                        if person1 in sent and person2 in sent:
                            relevant_sentence = sent.strip()
                            break
                    if not relevant_sentence and sentences:
                        relevant_sentence = sentences[0].strip()[:100]  # 如果没有找到，使用第一句
                    
                    paragraphs_data_for_excel.append((paragraph, line_idx, person1, person2, relevant_sentence))
    
    # 暂时关闭 LLM 分析，只导出段落数据
    print(f"\n⚠️ LLM 分析已暂时关闭，仅导出段落数据用于检查")
    print(f"✅ 准备导出 {len(paragraphs_data_for_excel)} 条段落记录到 Excel")
    
    # 返回空关系列表，但保留段落数据
    # LLM 分析已暂时关闭，只导出段落数据用于检查
    relationships = []
    all_names = set(names_list)
    
    # LLM 分析代码已暂时关闭，如需启用请取消注释以下代码
    # 注意：取消注释时需要确保三引号字符串正确配对
    # 
    # # 阶段3: 使用 LLM 分析段落
    # print(f"\n🔍 阶段3: 使用 LLM 分析段落中的人物关系...")
    # 
    # # 构建提示词模板（注意：使用双花括号 {{ 和 }} 来转义 JSON 示例中的花括号）
    # prompt_template = """你是一个专业的小说分析助手。请从以下文本段落中提取人物关系。
    # 
    # 要求：
    # 1. 识别段落中出现的所有人物姓名
    # 2. 提取人物之间的关系（如：父子、朋友、恋人、同事、敌人、师生、主仆、兄弟、姐妹等）
    # 3. 如果关系不明确，使用"相关"作为关系类型
    # 4. 只提取明确出现的关系，不要推测
    # 
    # 输出格式为 JSON 数组，每个元素格式如下：
    # {{
    #   "person1": "人物1",
    #   "relation": "关系类型",
    #   "person2": "人物2"
    # }}
    # 
    # 文本段落：
    # {text}
    # 
    # 请只返回 JSON 数组，不要包含其他解释文字。如果文本中没有人物关系，返回空数组 []。"""
    # 
    # # 分批处理段落
    # batch_size = 5  # 每批处理5个段落
    # for i in tqdm(range(0, len(unique_paragraphs), batch_size), desc="分析段落"):
    #     batch_paragraphs = unique_paragraphs[i:i+batch_size]
    #     
    #     # 合并多个段落为一个请求
    #     combined_text = "\n\n---\n\n".join([p[0] for p in batch_paragraphs])
    #     prompt = prompt_template.format(text=combined_text)
    #     
    #     try:
    #         # 调用 DeepSeek API
    #         response = client.chat.completions.create(
    #             model=model_name,
    #             messages=[
    #                 {"role": "user", "content": prompt}
    #             ],
    #             max_tokens=2000,
    #             temperature=0.3
    #         )
    #         
    #         # 解析响应（处理 reasoning_content 字段）
    #         message = response.choices[0].message
    #         content = message.content
    #         
    #         # 如果使用 deepseek-reasoner，可能需要处理 reasoning_content
    #         if hasattr(message, 'reasoning_content') and message.reasoning_content:
    #             # 只使用最终的 content，忽略思维链
    #             pass
    #         
    #         # 提取 JSON 数组
    #         json_match = re.search(r'\[.*\]', content, re.DOTALL)
    #         if json_match:
    #             json_str = json_match.group(0)
    #             try:
    #                 relations = json.loads(json_str)
    #                 for rel in relations:
    #                     if isinstance(rel, dict) and 'person1' in rel and 'person2' in rel:
    #                         person1 = rel['person1'].strip()
    #                         person2 = rel['person2'].strip()
    #                         relation = rel.get('relation', '相关').strip()
    #                         
    #                         # 过滤掉空名字
    #                         if not person1 or not person2:
    #                             continue
    #                         
    #                         # 添加人名到集合（允许 LLM 识别新的人名）
    #                         all_names.add(person1)
    #                         all_names.add(person2)
    #                         
    #                         # 记录关系（允许记录所有人名关系，不限制在原始列表中）
    #                         relationships.append((person1, relation, person2, 1.0))
    #             except json.JSONDecodeError as e:
    #                 print(f"⚠️ JSON 解析失败: {e}")
    #                 print(f"   响应内容: {content[:200]}")
    #         
    #         # 避免请求过快
    #         time.sleep(0.5)
    #         
    #     except Exception as e:
    #         print(f"⚠️ API 调用失败: {e}")
    #         continue
    # 
    # print(f"✅ 提取到 {len(relationships)} 个关系")
    
    return relationships, all_names, paragraphs_data_for_excel


def build_relation_matrix_from_llm(relationships, names_list):
    """
    从 LLM 分析结果构建关系矩阵
    
    Args:
        relationships: [(person1, relation, person2, weight), ...]
        names_list: 所有人名列表
    
    Returns:
        rel_matrix: 关系矩阵 (numpy array)
        names_array: 人名数组 (numpy array)
    """
    # 创建人名到索引的映射
    name_to_idx = {name: idx for idx, name in enumerate(names_list)}
    n = len(names_list)
    
    # 初始化关系矩阵
    rel_matrix = np.zeros((n, n))
    
    # 填充关系矩阵
    for person1, relation, person2, weight in relationships:
        if person1 in name_to_idx and person2 in name_to_idx:
            idx1 = name_to_idx[person1]
            idx2 = name_to_idx[person2]
            # 关系矩阵是对称的
            rel_matrix[idx1][idx2] = weight
            rel_matrix[idx2][idx1] = weight
    
    # 对角线存储每个名字的出现次数（用于排序）
    for i, name in enumerate(names_list):
        # 统计该名字在关系中的出现次数
        count = sum(1 for r in relationships if r[0] == name or r[2] == name)
        rel_matrix[i][i] = count
    
    return rel_matrix, np.array(names_list)


def export_paragraphs_to_excel(paragraphs_data, file_path, book_name=None):
    """
    导出找到的段落到 Excel 文件
    
    Args:
        paragraphs_data: 段落数据列表，每个元素为 (paragraph, line_idx, person1, person2, sentence)
        file_path: Excel 文件路径
        book_name: 书名（用于文件名）
    """
    if not PANDAS_AVAILABLE:
        print("⚠️ pandas 未安装，跳过段落导出")
        return
    
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        # 整理段落数据并去重
        paragraph_records = []
        seen_paragraphs = set()  # 用于去重段落内容
        
        for idx, (paragraph, line_idx, person1, person2, sentence) in enumerate(paragraphs_data, 1):
            # 使用段落内容作为唯一键进行去重
            paragraph_key = paragraph.strip()
            
            # 如果段落已存在，合并人名对信息（但不在Excel中重复显示）
            # 为了简化，我们只保留第一次出现的段落
            if paragraph_key in seen_paragraphs:
                continue  # 跳过重复段落
            
            seen_paragraphs.add(paragraph_key)
            
            paragraph_records.append({
                "序号": len(paragraph_records) + 1,  # 使用实际记录数，而不是原始idx
                "行号": line_idx + 1,  # 转换为 1-based 行号
                "人物1": person1,
                "人物2": person2,
                "包含的句子": sentence,
                "段落内容": paragraph,
                "段落长度": len(paragraph),
                "句子长度": len(sentence)
            })
        
        print(f"📊 去重后保留 {len(paragraph_records)} 条唯一段落记录（原始 {len(paragraphs_data)} 条）")
        
        df_paragraphs = pd.DataFrame(paragraph_records)
        
        # 写入 Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_paragraphs.to_excel(writer, sheet_name='找到的段落', index=False)
            
            # 调整列宽（如果可能）
            try:
                worksheet = writer.sheets['找到的段落']
                # 设置列宽
                worksheet.column_dimensions['A'].width = 8   # 序号
                worksheet.column_dimensions['B'].width = 10  # 行号
                worksheet.column_dimensions['C'].width = 15  # 人物1
                worksheet.column_dimensions['D'].width = 15  # 人物2
                worksheet.column_dimensions['E'].width = 50  # 包含的句子
                worksheet.column_dimensions['F'].width = 80  # 段落内容
                worksheet.column_dimensions['G'].width = 12  # 段落长度
                worksheet.column_dimensions['H'].width = 12  # 句子长度
            except Exception:
                pass  # 如果调整列宽失败，继续执行
        
        print(f"✅ 已导出段落到 Excel 文件: {file_path}")
        print(f"   - 共 {len(paragraph_records)} 个段落")
    except Exception as e:
        print(f"⚠️ 段落 Excel 导出失败: {e}")


def export_llm_relationships_to_excel(relationships, names_list, file_path, book_name=None):
    """
    导出 LLM 分析的关系到 Excel 文件
    
    Args:
        relationships: [(person1, relation, person2, weight), ...]
        names_list: 所有人名列表
        file_path: Excel 文件路径
        book_name: 书名（用于文件名）
    """
    if not PANDAS_AVAILABLE:
        print("⚠️ pandas 未安装，跳过 Excel 导出")
        return
    
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        # 关系详情表
        rel_data = []
        for person1, relation, person2, weight in relationships:
            rel_data.append({
                "人物1": person1,
                "关系": relation,
                "人物2": person2,
                "权重": weight
            })
        df_rel = pd.DataFrame(rel_data)
        
        # 人物统计表
        entity_data = []
        for name in names_list:
            count = sum(1 for r in relationships if r[0] == name or r[2] == name)
            entity_data.append({
                "人物": name,
                "关系数量": count
            })
        df_entity = pd.DataFrame(entity_data)
        df_entity = df_entity.sort_values("关系数量", ascending=False)
        
        # 关系类型统计表
        rel_type_counts = defaultdict(int)
        for _, relation, _, _ in relationships:
            rel_type_counts[relation] += 1
        rel_type_data = [{"关系类型": k, "数量": v} 
                        for k, v in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)]
        df_rel_type = pd.DataFrame(rel_type_data)
        
        # 写入 Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_rel.to_excel(writer, sheet_name='关系详情', index=False)
            df_entity.to_excel(writer, sheet_name='人物统计', index=False)
            df_rel_type.to_excel(writer, sheet_name='关系类型统计', index=False)
        
        print(f"✅ 已导出 Excel 文件: {file_path}")
    except Exception as e:
        print(f"⚠️ Excel 导出失败: {e}")


def sanitize_filename(filename):
    """清理文件名，移除或替换不允许的字符"""
    import re
    # 移除或替换文件系统不支持的字符
    # Windows 不支持的字符: < > : " / \ | ? *
    # 保留中文字符、字母、数字、下划线、连字符、空格
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除首尾空格和点
    filename = filename.strip(' .')
    # 限制文件名长度（避免过长）
    if len(filename) > 100:
        filename = filename[:100]
    return filename

def plot_rel(relations, names, draw_all=True, balanced=True, verbose=True, save_path=None, book_name=None):

    # 平衡名字关系
    if balanced == True:
        relations =(relations.T+relations)/2
    

    # 画图
    G = nx.Graph()

    # 将每个名字，和名字出现的次数加入图
    nums = np.diag(relations)
    for i,name in enumerate(names):
        G.add_node(name, num = nums[i])

    # 将关系加入图
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if relations[i, j] != 0:
                G.add_edge(names[i], names[j], weight=relations[i, j])

    # 判断是否联通并切分子图
    max_weight = 0.0
    #### for c in sorted(nx.connected_components(G), key=len, reverse=True):
    #　画出主要子图
    main_c = max(nx.connected_components(G), key=len)
    sub_G = G.subgraph(main_c)
    sub_nums = np.array([n[1] for n in sub_G.nodes(data="num")])
    sub_weight = np.array([e[2] for e in sub_G.edges(data="weight")])
    if len(sub_weight) != 0:  # 权重值为 0 则不需要归一化
        max_weight = max(np.max(sub_weight), max_weight)
        sub_weight = sub_weight*4.5/max_weight

    #主要子图外其他的图
    other_c = set(G.nodes) - main_c

    #最终结果信息
    info = "<<shown-points>>\n{}\n<<dropout-points>>\n{}".format(
        sub_G.nodes(data="num"), G.subgraph(other_c).nodes(data="num"))
    
    if verbose == True:
        print("="*50)
        print("+++++++ 最终分析结果: +++++++")
        print(info)
        print("="*50)

    # 检测是否在 CI 环境或需要保存文件
    is_ci = os.getenv('CI') == 'true' or os.getenv('DISPLAY') is None
    save_images = save_path is not None or is_ci
    
    if save_images and save_path is None:
        save_path = "output"
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # 根据节点数量调整参数以避免标签重叠
    num_nodes = len(sub_G.nodes())
    
    # 动态调整参数 - 增大画布以容纳更多内容
    if num_nodes > 50:
        # 大量节点时：增大画布、减小字体、增大节点间距
        figsize = (32, 24)  # 从 (20, 16) 增大到 (32, 24)
        font_size = 6
        node_size_multiplier = 20
        k_value = 3  # 用于 spring 布局的节点间距
    elif num_nodes > 30:
        # 中等节点时
        figsize = (24, 20)  # 从 (16, 14) 增大到 (24, 20)
        font_size = 8
        node_size_multiplier = 40
        k_value = 2
    else:
        # 少量节点时
        figsize = (18, 15)  # 从 (12, 10) 增大到 (18, 15)
        font_size = 10
        node_size_multiplier = 60
        k_value = 1
    
    # 调整节点大小（确保最小尺寸）
    node_sizes = np.maximum(sub_nums * node_size_multiplier, 50)
    
    #多种方式展示结果
    def spring_layout_func(G):
        return nx.spring_layout(G, k=k_value, iterations=50)
    
    layouts = [
        ("spring", spring_layout_func),
        ("circular", nx.circular_layout),
        ("kamada_kawai", nx.kamada_kawai_layout),
        ("spectral", nx.spectral_layout),
        ("random", nx.random_layout)
    ]
    
    layout_count = len(layouts) if draw_all else 1
    
    for i, (layout_name, layout_func) in enumerate(layouts[:layout_count]):
        try:
            plt.figure(figsize=figsize)
            
            # 计算布局位置（添加超时处理）
            try:
                pos = layout_func(sub_G)
            except Exception as e:
                print(f"⚠️ 布局算法 {layout_name} 计算失败: {e}")
                print(f"   使用 spring 布局作为备选")
                pos = spring_layout_func(sub_G)
            
            # 绘制节点和边
            nx.draw_networkx_nodes(sub_G, pos, node_size=node_sizes, node_color='lightblue', 
                                  alpha=0.7, edgecolors='black', linewidths=0.5)
            nx.draw_networkx_edges(sub_G, pos, width=sub_weight, alpha=0.5, edge_color='gray')
            
            # 绘制标签，使用更好的参数避免重叠
            labels = {node: node for node in sub_G.nodes()}
            nx.draw_networkx_labels(sub_G, pos, labels, font_size=font_size, 
                                   font_family='sans-serif', 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            edgecolor='none', alpha=0.7))
            
            plt.title(f"人物关系图 - {layout_name} (共{num_nodes}个人物)", fontsize=14, pad=20)
            plt.axis('off')
            
            if save_images:
                # 生成文件名，包含书籍名称（如果提供）
                if book_name:
                    safe_book_name = sanitize_filename(book_name)
                    filename_base = f"{safe_book_name}_relationship_{layout_name}"
                else:
                    filename_base = f"relationship_{layout_name}"
                
                filename = os.path.join(save_path, f"{filename_base}.png") if save_path else f"{filename_base}.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                if verbose:
                    print(f"✅ 已保存图片: {filename} (节点数: {num_nodes})")
            plt.close()
        except Exception as e:
            print(f"❌ 生成 {layout_name} 布局图时出错: {e}")
            if 'plt' in locals():
                plt.close()
            continue
    # nx.draw_shell(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    # plt.show()

def trans_list2dict(trans_list):
    """
    把别名转换列表转换为别名转换字典
    """
    trans_dict = {}
    for names in trans_list:
        for i,name in enumerate(names):
            if i==0:
                continue
            trans_dict[name] = names[0]
    return trans_dict



# ["罗辑","程心","汪淼","叶文洁","史强","维德","云天明","希恩斯","雷迪亚兹","丁仪","泰勒","章北海","关一帆","文洁","北海","天明","一帆","伟思","文斯","卫宁","始皇","心说","文王","玉菲","志成","西里","晓明","哲泰","庄颜","墨子","杨晋文","晋文","慈欣","沐霖","张援朝","援朝","艾AA","AA"]
# info = ["林黛玉","薛宝钗","贾元春","贾迎春","贾探春","贾惜春","李纨","妙玉","史湘云","王熙凤","贾巧姐","秦可卿","晴雯","麝月","袭人","鸳鸯","雪雁","紫鹃","碧痕","平儿","香菱","金钏","司棋","抱琴","赖大","焦大","王善保","周瑞","林之孝","乌进孝","包勇","吴贵","吴新登","邓好时","王柱儿","余信","庆儿","昭儿","兴儿","隆儿","坠儿","喜儿","寿儿","丰儿","住儿","小舍儿","李十儿","玉柱儿","贾敬","贾赦","贾政","贾宝玉","贾琏","贾珍","贾环","贾蓉","贾兰","贾芸","贾蔷","贾芹","琪官","芳官","藕官","蕊官","药官","玉官","宝官","龄官","茄官","艾官","豆官","葵官","妙玉","智能","智通","智善","圆信","大色空","净虚","彩屏","彩儿","彩凤","彩霞","彩鸾","彩明","彩云","贾元春","贾迎春","贾探春","贾惜春","薛蟠","薛蝌","薛宝钗","薛宝琴","王夫人","王熙凤","王子腾","王仁","尤老娘","尤氏","尤二姐","尤三姐","贾蓉","贾兰","贾芸","贾芹","贾珍","贾琏","贾环","贾瑞","贾敬","贾赦","贾政","贾敏","贾代儒","贾代化","贾代修","贾代善","晴雯","金钏","鸳鸯","司棋","詹光","单聘仁","程日兴","王作梅","石呆子","张华","冯渊","张金哥","茗烟","扫红","锄药","伴鹤","小鹊","小红","小蝉","小舍儿","刘姥姥","马道婆","宋嬷嬷","张妈妈","秦锺","蒋玉菡","柳湘莲","东平王","乌进孝","冷子兴","山子野","方椿","载权","夏秉忠","周太监","裘世安","抱琴","司棋","侍画","入画","珍珠","琥珀","玻璃","翡翠","史湘云","翠缕","笑儿","篆儿贾探春","侍画","翠墨","小蝉","贾宝玉","茗烟","袭人","晴雯","林黛玉","紫鹃","雪雁","春纤","贾惜春","入画","彩屏","彩儿","贾迎春","彩凤","彩云","彩霞"] 
# hanlp.add(info)
parser = argparse.ArgumentParser(description="指定书的名字")

parser.add_argument("--book", default="weicheng", type=str,
                    help="书的名字，不带后缀")
parser.add_argument("--debug",default=False,type=bool,help="控制中间结果的输出。默认关闭")
# LLM 分析相关参数（默认使用 LLM）
parser.add_argument("--use_cooccurrence", action="store_true",
                    help="使用共现统计方法，而不是 LLM 分析（默认使用 LLM）")
parser.add_argument("--api_base_url", type=str, default=None,
                    help="API 基础 URL（默认从环境变量 API_BASE_URL 读取，或使用 https://api.deepseek.com）")
parser.add_argument("--api_key", type=str, default=None,
                    help="API 密钥（默认从环境变量 API_KEY 读取）")
parser.add_argument("--model", type=str, default=None,
                    help="模型名称（默认从环境变量 API_MODEL 读取，或使用 deepseek-reasoner）")
parser.add_argument("--max_sentences", type=int, default=200,
                    help="最多分析的句子数（默认 200）")
parser.add_argument("--context_lines", type=int, default=3,
                    help="段落上下文行数（默认 3）")

if __name__ == "__main__":

    # a = str(CustomDictionary.get("鸿渐"))
    # print(a=="nz 3 ")
    #################################################
    # ############################################# 
    # ############# 手动调整模型 ####################
    # 前期添加的字典
    name_dict = []
    
    # 后期效果优化
    trans_list = [] 
    # 转换列表，格式如下
    # [[name1,name1_,...],[name2,name2_,...],... ]
    # 列表内的每一个列表代表一个人物的一组别名，所有别名会转换为第一个名字
    
    trans_dict = {}
    trans_dict.update(trans_list2dict(trans_list))

    err_list = []

    threshold = -1
    # ############################################
    # ############################################
    
    # 获取书名参数
    args = parser.parse_args()
    fp = "book/"+ args.book +".txt"
    assert os.path.exists(fp),"error!: no such book in "+ fp
    print("=====+++=== NER for book: "+fp+" ===+++=====",flush=True)
    ###################################33
    ###############################
    # 插入个性化字典
    # name_dict = []
    hanlp.add(name_dict)
    #################################
    #################################
    
    # 感知机分析器对文本进行分析
    model = hanlp(custom_dict=True)
    rels, ns, nr_nrf_dict = count_names(fp, model)
    if args.debug:
        f = np.diag(rels) >= 40
        print("="*50)
        print("<<粗提取结果>>\n名字总数: {} \n{}{}".format(len(ns),ns[f],np.diag(rels)[f]))
        print("="*50)

    ## 分别生成新的名称字典，以及转换字典
    # print(filter_nr(nr_nrf_dict))
    auto_name_list, auto_trans_dict = filter_nr(nr_nrf_dict,first=True)
    if args.debug:
        print("="*50)
        print("<<自动生成的名称列表和名称转换字典>>")
        print("名称列表:\n", auto_name_list)
        print("名称转换字典\n",auto_trans_dict)
        print("="*50)
    hanlp.add(auto_name_list)
    
          
    ############################################
    # 手动调整的转换字典
    auto_trans_dict.update(trans_dict)
    trans_dict = auto_trans_dict
    ###############################################
    
    
    # 默认使用 LLM 分析，除非明确指定使用共现统计
    use_llm = not args.use_cooccurrence
    
    # 先进行共现统计，获取最终过滤后的人名列表（用于段落查找）
    print(f"\n{'='*60}")
    print(f"第一步：共现统计（获取最终人名列表）")
    print(f"{'='*60}")
    
    ### 重新进行统计和计数（使用已添加的字典）
    model = hanlp(custom_dict=True)#,analyzer="CRF")
    rels,ns,_ = count_names(fp,model)
  
    ##### 根据手工调整以不同效果展示
    relations_cooccurrence, names_cooccurrence = filter_names(
            rels, ns, trans=trans_dict, err=err_list, threshold=threshold)
    
    # 获取最终的人名列表（用于段落查找）
    final_names_list = list(names_cooccurrence)
    
    # 过滤掉明显不是人名的词
    def filter_non_person_names(names):
        """过滤掉明显不是人名的词"""
        # 明显不是人名的词列表
        exclude_words = {
            '闻言', '披萨', '福克斯', '王',  # 明显不是人名
            '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一',  # 常见词
            '这个', '那个', '什么', '怎么', '为什么', '可以', '不能'
        }
        filtered = [name for name in names if name not in exclude_words]
        return filtered
    
    # 过滤非人名
    final_names_list = filter_non_person_names(final_names_list)
    print(f"\n✅ 共现统计完成，得到 {len(final_names_list)} 个最终人名（已过滤非人名）")
    print(f"   人名列表: {final_names_list}")
    if len(names_cooccurrence) != len(final_names_list):
        removed = set(names_cooccurrence) - set(final_names_list)
        print(f"   已排除的非人名: {sorted(removed)}")
    
    if use_llm:
        # 使用 LLM 分析（默认模式）
        api_key = args.api_key or os.getenv('API_KEY')
        api_base_url = args.api_base_url or os.getenv('API_BASE_URL', 'https://api.deepseek.com')
        model_name = args.model or os.getenv('API_MODEL', 'deepseek-reasoner')
        
        if not api_key:
            print("⚠️ 警告: 未提供 API 密钥，无法使用 LLM 分析")
            print("   回退到共现统计方法")
            print("   提示: 设置环境变量 API_KEY 或使用 --api_key 参数以启用 LLM 分析")
            use_llm = False
        
        if use_llm:
            print(f"\n{'='*60}")
            print(f"第二步：LLM 分析模式（使用最终人名列表）")
            print(f"{'='*60}")
        
            # 读取文本文件
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text_lines = [line.strip() for line in f.readlines() if line.strip()]
            except UnicodeDecodeError:
                with open(fp, "r", encoding="gbk") as f:
                    text_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"📖 文本文件: {fp}")
            print(f"📝 总共有 {len(text_lines)} 行文本")
            print(f"📡 API 地址: {api_base_url}")
            print(f"📦 模型: {model_name}")
            
            # 使用最终过滤后的人名列表进行 LLM 分析（而不是36个高频人名）
            print(f"\n📋 使用共现统计过滤后的 {len(final_names_list)} 个最终人名进行段落查找")
            
            # 调用 LLM 分析函数
            relationships, all_names, paragraphs_data = analyze_relationships_with_llm(
                text_lines,
                final_names_list,  # 使用最终过滤后的人名列表
                base_url=api_base_url,
                api_key=api_key,
                model_name=model_name,
                max_sentences=args.max_sentences,
                context_lines=args.context_lines
            )
            
            # 导出段落数据（无论是否有关系，因为LLM已关闭）
            if PANDAS_AVAILABLE and paragraphs_data:
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                paragraphs_excel_path = os.path.join(output_dir, f"{sanitize_filename(args.book)}_找到的段落.xlsx")
                export_paragraphs_to_excel(paragraphs_data, paragraphs_excel_path, args.book)
            
            if len(relationships) == 0:
                print("⚠️ LLM 未提取到任何关系（LLM 分析已关闭）")
                print("💡 段落数据已导出到 Excel，请检查内容是否正确")
                use_llm = False
                # 回退到共现统计结果
                relations = relations_cooccurrence
                names = names_cooccurrence
            else:
                # 构建关系矩阵
                # 合并所有名字，优先使用 final_names_list 中的顺序
                all_names_list = list(all_names)
                # 先按 final_names_list 的顺序排序，然后加上不在列表中的名字
                names_in_list = [name for name in final_names_list if name in all_names_list]
                names_not_in_list = [name for name in all_names_list if name not in final_names_list]
                names_list_sorted = names_in_list + names_not_in_list
                
                relations, names = build_relation_matrix_from_llm(relationships, names_list_sorted)
                
                print(f"\n✅ LLM 分析完成，提取到 {len(relationships)} 个关系")
                
                # 导出关系数据（如果可用）
                if PANDAS_AVAILABLE:
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    excel_path = os.path.join(output_dir, f"{sanitize_filename(args.book)}_人物关系_LLM.xlsx")
                    export_llm_relationships_to_excel(relationships, names_list_sorted, excel_path, args.book)
    
    if not use_llm:
        # 使用原有的共现统计方法
        print(f"\n{'='*60}")
        print(f"使用共现统计模式")
        print(f"{'='*60}")
        
        # 使用之前已经计算好的结果
        relations = relations_cooccurrence
        names = names_cooccurrence

    ##### 展示最终结果和信息
    # 传递书籍名称给 plot_rel 函数，用于生成带书籍名的文件名
    plot_rel(relations, names, book_name=args.book)

   
