# encoding=utf-8
# 人物关系抽取模块 - 基于规则和关键词的 Hybrid 方法
# 无需训练模型，可直接使用

import re
from typing import List, Dict, Tuple, Set
import numpy as np


class RelationExtractor:
    """基于规则和关键词的人物关系抽取器"""
    
    def __init__(self):
        """初始化关系关键词词典"""
        # 关系关键词词典 - 按类别组织
        self.relation_patterns = {
            # 家庭关系
            'family': {
                '父子': ['父亲', '爸爸', '爸', '爹', '儿子', '儿子', '子', '父子', '父子关系'],
                '母子': ['母亲', '妈妈', '妈', '娘', '母子', '母子关系'],
                '父女': ['父亲', '爸爸', '爸', '爹', '女儿', '闺女', '父女', '父女关系'],
                '母女': ['母亲', '妈妈', '妈', '娘', '女儿', '闺女', '母女', '母女关系'],
                '夫妻': ['妻子', '老婆', '夫人', '太太', '媳妇', '内人', 
                        '丈夫', '老公', '先生', '相公', '夫君', '夫妻', '夫妻关系', '夫妇'],
                '兄弟': ['哥哥', '弟弟', '兄', '弟', '兄弟', '弟兄', '兄长', '弟弟'],
                '姐妹': ['姐姐', '妹妹', '姐', '妹', '姐妹', '姊妹'],
                '兄妹': ['哥哥', '兄', '妹妹', '妹', '兄妹'],
                '姐弟': ['姐姐', '姐', '弟弟', '弟', '姐弟'],
                '祖孙': ['爷爷', '奶奶', '祖父', '祖母', '外公', '外婆', 
                        '孙子', '孙女', '外孙', '外孙女', '祖孙', '祖孙关系'],
                '叔侄': ['叔叔', '伯伯', '伯父', '叔父', '侄子', '侄女', '叔侄'],
                '舅甥': ['舅舅', '舅父', '外甥', '外甥女', '舅甥'],
            },
            # 社会关系
            'social': {
                '朋友': ['朋友', '好友', '伙伴', '伙伴', '兄弟', '闺蜜', '死党', '挚友', '知己'],
                '师生': ['老师', '教师', '师父', '师傅', '恩师', '学生', '徒弟', '弟子', '门生'],
                '同学': ['同学', '同窗', '校友', '同班', '同校'],
                '同事': ['同事', '同僚', '伙伴', '同事', '同工'],
                '上下级': ['上司', '领导', '老板', '上级', '主管', '下属', '员工', '手下', '部下'],
                '主仆': ['主人', '主子', '仆人', '佣人', '下人', '家丁', '丫鬟'],
            },
            # 情感关系
            'emotion': {
                '恋人': ['恋人', '情侣', '男朋友', '女朋友', '爱人', '对象', '心上人', '意中人','对象','爱侣','亲爱的'],
                '仇敌': ['敌人', '仇人', '对手', '对头', '仇敌', '死敌'],
                '恩人': ['恩人', '救命恩人', '恩公'],
                '仇人': ['仇人', '仇家', '死对头'],
            },
            # 其他关系
            'other': {
                '师徒': ['师父', '师傅', '徒弟', '弟子', '徒儿'],
                '主从': ['主人', '从', '随从', '跟班'],
                '盟友': ['盟友', '同盟', '伙伴', '战友'],
            }
        }
        
        # 关系指示词（用于判断关系的方向）
        self.relation_indicators = {
            '的': ['父亲', '母亲', '儿子', '女儿', '朋友', '老师', '学生'],
            '和': ['和', '与', '同', '跟', '及'],
            '是': ['是', '为', '乃'],
            '叫': ['叫', '称', '称呼'],
        }
        
        # 否定词（用于排除非关系）
        self.negation_words = ['不是', '非', '没有', '无', '不']
    
    def extract_relations(self, text: str, persons: List[str], 
                         window_size: int = 200) -> List[Dict]:
        """
        从文本中抽取人物关系
        
        Args:
            text: 输入文本
            persons: 人物列表
            window_size: 滑动窗口大小（字符数），用于处理长文本
        
        Returns:
            关系列表，每个关系包含 person1, person2, relation, confidence 等信息
        """
        if not persons or len(persons) < 2:
            return []
        
        all_relations = []
        relations_set = set()  # 用于去重
        
        # 使用滑动窗口处理长文本，提高效率
        if len(text) > window_size * 2:
            for i in range(0, len(text), window_size):
                end_idx = min(i + window_size * 2, len(text))
                window_text = text[i:end_idx]
                relations = self._extract_from_window(window_text, persons)
                all_relations.extend(relations)
        else:
            relations = self._extract_from_window(text, persons)
            all_relations.extend(relations)
        
        # 去重：相同的人物对和关系类型只保留一次
        unique_relations = []
        for rel in all_relations:
            key = (rel['person1'], rel['person2'], rel['relation'])
            if key not in relations_set:
                relations_set.add(key)
                unique_relations.append(rel)
        
        return unique_relations
    
    def _extract_from_window(self, text: str, persons: List[str]) -> List[Dict]:
        """从文本窗口提取关系"""
        relations = []
        
        # 查找句子中的人物
        found_persons = []
        person_positions = {}
        for person in persons:
            # 查找人物的所有出现位置
            positions = []
            for match in re.finditer(re.escape(person), text):
                positions.append(match.start())
            if positions:
                found_persons.append(person)
                person_positions[person] = positions
        
        if len(found_persons) < 2:
            return relations
        
        # 分句处理（按句号、问号、感叹号分句）
        sentences = re.split(r'[。！？\n]', text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 检查句子中是否包含多个人物
            sentence_persons = [p for p in found_persons if p in sentence]
            
            if len(sentence_persons) < 2:
                continue
            
            # 方法1: 基于关键词匹配
            relations.extend(self._match_by_keywords(sentence, sentence_persons))
            
            # 方法2: 基于模式匹配（如 "A是B的X"）
            relations.extend(self._match_by_patterns(sentence, sentence_persons))
            
            # 方法3: 基于共现和距离（两人距离很近且有关系词）
            relations.extend(self._match_by_proximity(sentence, sentence_persons))
        
        return relations
    
    def _match_by_keywords(self, sentence: str, persons: List[str]) -> List[Dict]:
        """基于关键词匹配关系"""
        relations = []
        
        # 遍历所有关系类型
        for category, rels in self.relation_patterns.items():
            for rel_type, keywords in rels.items():
                for keyword in keywords:
                    if keyword in sentence:
                        # 找到关系关键词，关联句子中的所有人
                        for i, p1 in enumerate(persons):
                            for j, p2 in enumerate(persons):
                                if i != j:
                                    # 计算置信度：关键词距离人物的远近
                                    confidence = self._calculate_confidence(
                                        sentence, keyword, p1, p2
                                    )
                                    
                                    relations.append({
                                        'person1': p1,
                                        'person2': p2,
                                        'relation': rel_type,
                                        'category': category,
                                        'keyword': keyword,
                                        'confidence': confidence,
                                        'method': 'keyword_match',
                                        'sentence': sentence[:100]  # 截断长句子
                                    })
        
        return relations
    
    def _match_by_patterns(self, sentence: str, persons: List[str]) -> List[Dict]:
        """基于模式匹配（如 "A是B的父亲"）"""
        relations = []
        
        # 常见的关系模式
        patterns = [
            # "A是B的X"
            (r'([^，。！？\s]+)是([^，。！？\s]+)的([^，。！？\s]+)', 1, 2, 3),
            # "A和B是X"
            (r'([^，。！？\s]+)和([^，。！？\s]+)是([^，。！？\s]+)', 1, 2, 3),
            # "A称B为X"
            (r'([^，。！？\s]+)称([^，。！？\s]+)为([^，。！？\s]+)', 1, 2, 3),
        ]
        
        for pattern, p1_idx, p2_idx, rel_idx in patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                groups = match.groups()
                if len(groups) >= max(p1_idx, p2_idx, rel_idx):
                    p1_candidate = groups[p1_idx - 1]
                    p2_candidate = groups[p2_idx - 1]
                    rel_candidate = groups[rel_idx - 1]
                    
                    # 检查是否在人物列表中
                    p1 = self._match_person(p1_candidate, persons)
                    p2 = self._match_person(p2_candidate, persons)
                    
                    if p1 and p2 and p1 != p2:
                        # 检查关系词是否匹配
                        rel_type = self._match_relation_type(rel_candidate)
                        if rel_type:
                            relations.append({
                                'person1': p1,
                                'person2': p2,
                                'relation': rel_type,
                                'confidence': 0.8,
                                'method': 'pattern_match',
                                'sentence': sentence[:100]
                            })
        
        return relations
    
    def _match_by_proximity(self, sentence: str, persons: List[str]) -> List[Dict]:
        """基于人物距离和关系词的共现"""
        relations = []
        
        if len(persons) < 2:
            return relations
        
        # 计算人物之间的距离
        for i, p1 in enumerate(persons):
            for j, p2 in enumerate(persons):
                if i >= j:
                    continue
                
                # 查找两个人之间的最小距离
                p1_pos = sentence.find(p1)
                p2_pos = sentence.find(p2)
                
                if p1_pos == -1 or p2_pos == -1:
                    continue
                
                distance = abs(p1_pos - p2_pos)
                
                # 如果距离很近（小于50个字符），检查是否有关系词
                if distance < 50:
                    # 提取两个人之间的文本
                    start = min(p1_pos, p2_pos)
                    end = max(p1_pos + len(p1), p2_pos + len(p2))
                    between_text = sentence[start:end]
                    
                    # 查找关系词
                    for category, rels in self.relation_patterns.items():
                        for rel_type, keywords in rels.items():
                            for keyword in keywords:
                                if keyword in between_text:
                                    confidence = max(0.4, 0.8 - distance / 100)
                                    relations.append({
                                        'person1': p1,
                                        'person2': p2,
                                        'relation': rel_type,
                                        'category': category,
                                        'confidence': confidence,
                                        'method': 'proximity',
                                        'distance': distance,
                                        'sentence': sentence[:100]
                                    })
                                    break
        
        return relations
    
    def _match_person(self, candidate: str, persons: List[str]) -> str:
        """匹配候选字符串是否为人名"""
        # 精确匹配
        if candidate in persons:
            return candidate
        
        # 部分匹配（如果候选字符串包含人名）
        for person in persons:
            if person in candidate or candidate in person:
                return person
        
        return None
    
    def _match_relation_type(self, rel_candidate: str) -> str:
        """匹配关系类型"""
        for category, rels in self.relation_patterns.items():
            for rel_type, keywords in rels.items():
                if rel_candidate in keywords:
                    return rel_type
        return None
    
    def _calculate_confidence(self, sentence: str, keyword: str, 
                             p1: str, p2: str) -> float:
        """计算关系置信度"""
        base_confidence = 0.6
        
        # 如果关键词在两个人之间，置信度更高
        p1_pos = sentence.find(p1)
        p2_pos = sentence.find(p2)
        keyword_pos = sentence.find(keyword)
        
        if p1_pos != -1 and p2_pos != -1 and keyword_pos != -1:
            min_pos = min(p1_pos, p2_pos)
            max_pos = max(p1_pos + len(p1), p2_pos + len(p2))
            
            # 关键词在两个人物之间
            if min_pos < keyword_pos < max_pos:
                base_confidence = 0.8
            # 关键词距离人物很近
            elif abs(keyword_pos - min_pos) < 20 or abs(keyword_pos - max_pos) < 20:
                base_confidence = 0.7
        
        # 检查是否有否定词
        for neg_word in self.negation_words:
            if neg_word in sentence:
                base_confidence *= 0.3  # 大幅降低置信度
        
        return min(base_confidence, 1.0)
    
    def build_relation_matrix(self, relations: List[Dict], 
                            persons: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        构建关系矩阵和关系信息字典
        
        Args:
            relations: 关系列表
            persons: 人物列表
        
        Returns:
            (关系矩阵, 关系信息字典)
        """
        n = len(persons)
        matrix = np.zeros((n, n), dtype=np.float32)
        relation_info = {}  # 存储详细关系信息
        
        # 统计关系权重
        for rel in relations:
            p1 = rel.get('person1')
            p2 = rel.get('person2')
            
            if p1 and p2 and p1 in persons and p2 in persons:
                i = persons.index(p1)
                j = persons.index(p2)
                confidence = rel.get('confidence', 0.5)
                
                # 累加权重（对称矩阵）
                matrix[i, j] += confidence
                matrix[j, i] += confidence
                
                # 存储关系信息（取置信度最高的关系）
                key = tuple(sorted([p1, p2]))
                if key not in relation_info:
                    relation_info[key] = []
                relation_info[key].append(rel)
        
        # 对每个关系对，选择置信度最高的关系
        final_relation_info = {}
        for key, rels in relation_info.items():
            best_rel = max(rels, key=lambda x: x.get('confidence', 0))
            final_relation_info[key] = best_rel
        
        return matrix, final_relation_info
    
    def merge_relation_matrices(self, cooccurrence_matrix: np.ndarray,
                               semantic_matrix: np.ndarray,
                               alpha: float = 0.3) -> np.ndarray:
        """
        合并共现矩阵和语义关系矩阵
        
        Args:
            cooccurrence_matrix: 共现统计矩阵（原有方法）
            semantic_matrix: 语义关系矩阵（新增方法）
            alpha: 语义关系权重（0-1）
        
        Returns:
            合并后的关系矩阵
        """
        # 归一化矩阵
        if cooccurrence_matrix.max() > 0:
            cooccurrence_matrix = cooccurrence_matrix / cooccurrence_matrix.max()
        
        if semantic_matrix.max() > 0:
            semantic_matrix = semantic_matrix / semantic_matrix.max()
        
        # 合并：共现关系 + 语义关系
        merged = (1 - alpha) * cooccurrence_matrix + alpha * semantic_matrix
        
        return merged

