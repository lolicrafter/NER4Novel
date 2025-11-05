# 使用 PaddleNLP UIE 进行人物关系提取

## 简介

`rel_uie.py` 使用 PaddleNLP 的通用信息抽取（UIE）技术来提取小说中的人物关系。相比原有的 `rel.py`（基于 HanLP 的共现统计），UIE 可以：

1. **识别具体的关系类型**：如父子、朋友、同事、恋人等
2. **更准确的关系抽取**：基于语义理解，而非简单的共现统计
3. **导出 Excel 表格**：自动生成包含关系详情、人物统计、关系类型统计的工作表
4. **绘制关系图**：可视化人物关系网络

## 安装依赖

```bash
pip install paddlenlp>=2.5.0 pandas>=1.3.0 openpyxl>=3.0.0
```

或者安装所有依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python rel_uie.py --book 冬日重现
```

### 完整参数

```bash
python rel_uie.py \
    --book 冬日重现 \
    --batch_size 32 \
    --max_length 512 \
    --output output
```

### 参数说明

- `--book`: 书籍名称（不带 .txt 后缀），默认：`冬日重现`
- `--batch_size`: 批处理大小，默认：`32`
- `--max_length`: 最大文本长度，默认：`512`
- `--output`: 输出目录，默认：`output`
- `--schema`: 自定义 schema（JSON 格式），默认使用人物关系抽取

### 自定义 Schema（高级用法）

UIE 支持自定义 schema 来定义需要抽取的实体和关系。例如：

```bash
python rel_uie.py --book 冬日重现 --schema '{"人物": ["关系"]}'
```

## 输出结果

运行后会生成：

1. **关系图**：`output/冬日重现_uie_relationship.png`
   - 可视化的人物关系网络图
   - 节点大小表示人物重要性
   - 边的粗细表示关系强度

2. **Excel 文件**：`output/冬日重现_人物关系.xlsx`
   包含三个工作表：
   - **关系详情**：所有人物关系三元组（人物1, 关系类型, 人物2, 置信度）
   - **人物统计**：每个人物及其关系数量
   - **关系类型统计**：各种关系类型的出现次数

## 对比：UIE vs 原有方法

| 特性 | rel.py (HanLP) | rel_uie.py (UIE) |
|------|----------------|------------------|
| 人名识别 | ✅ 基于 NER | ✅ 基于 NER |
| 关系类型 | ❌ 仅共现 | ✅ 具体关系类型 |
| 关系准确性 | 中等 | 较高 |
| Excel 导出 | ❌ | ✅ |
| 关系可视化 | ✅ | ✅ |
| 处理速度 | 快 | 较慢（需要模型推理） |

## 注意事项

1. **首次运行**：UIE 模型会自动下载，可能需要一些时间
2. **内存需求**：UIE 模型需要较多内存，建议至少 4GB 可用内存
3. **文本长度**：过长的文本会被截断到 `max_length`，可以调整此参数
4. **关系提取质量**：取决于文本质量和模型能力，可能需要调整 schema 优化结果

## 常见问题

### Q: 提取不到关系怎么办？

A: 可以尝试：
1. 调整 `--max_length` 参数，增加文本长度
2. 自定义 schema，明确指定需要抽取的关系类型
3. 检查文本格式，确保是 UTF-8 编码

### Q: 模型下载失败？

A: 可以手动下载模型或使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 如何提高关系提取的准确性？

A: 
1. 使用更大的模型（如 `uie-large`）
2. 针对特定领域进行模型微调
3. 调整 schema，明确关系类型

## 示例

```bash
# 分析《冬日重现》
python rel_uie.py --book 冬日重现

# 使用更大的批处理大小
python rel_uie.py --book 冬日重现 --batch_size 64

# 自定义输出目录
python rel_uie.py --book 冬日重现 --output results
```

## 技术细节

- **UIE 模型**：使用 PaddleNLP 的 `uie-base` 模型
- **关系抽取**：基于 Schema 的零样本信息抽取
- **可视化**：使用 NetworkX 和 Matplotlib
- **数据导出**：使用 Pandas 和 openpyxl

