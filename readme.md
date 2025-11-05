# 小说人物关系提取

基于 HanLP 和 LLM API 的小说人物关系分析工具。

## 功能

- ✅ 使用 HanLP 提取小说中的人名
- ✅ 使用 LLM API 分析人物关系
- ✅ 自动生成人物关系图
- ✅ 导出 Excel 表格（关系详情、人物统计、关系类型统计）

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备小说文件
# 将小说 txt 文件放在 book/ 目录下

# 3. 运行分析（使用 HanLP 共现统计）
python rel.py --book 冬日重现

# 4. 运行分析（使用 LLM API，需要配置 API 密钥）
export API_BASE_URL="https://miaodi.zeabur.app"
export API_KEY="your-api-key"
export API_MODEL="deepseek-ai/DeepSeek-V3-0324"
python rel_api_optimized.py --book 冬日重现
```

### 输出结果

- **关系图**: `output/书名_relationship.png`
- **Excel 文件**: `output/书名_人物关系.xlsx`

## 脚本说明

| 脚本 | 说明 | 适用场景 |
|------|------|----------|
| `rel.py` | 使用 HanLP 共现统计 | 本地快速分析 |
| `rel_api_optimized.py` | 使用 LLM API 分析关系 | 需要准确关系类型 |

## GitHub Actions 配置

### 方式 1: Repository Secrets（推荐，简单）

在 GitHub 仓库的 **Settings → Secrets and variables → Actions → Secrets** 中添加：

| Secret 名称 | 值 | 说明 |
|------------|-----|------|
| `API_BASE_URL` | `https://miaodi.zeabur.app` | API 基础地址 |
| `API_KEY` | `your-api-key` | API 密钥 |
| `API_MODEL` | `deepseek-ai/DeepSeek-V3-0324` | 模型名称 |

**优点**: 配置简单，无需额外设置

### 方式 2: Environment Secrets（高级，适合多环境）

如果需要更细粒度的权限控制或多环境配置，可以使用 Environment Secrets：

#### 步骤 1: 创建 Environment

1. 进入 **Settings → Environments**
2. 点击 **New environment**
3. 输入环境名称（如 `production`、`staging`、`development`）
4. 点击 **Configure environment**

#### 步骤 2: 配置 Environment Secrets

在创建的环境页面中：
1. 滚动到 **Environment secrets** 部分
2. 点击 **Add secret**
3. 添加以下 secrets：
   - `API_BASE_URL`
   - `API_KEY`
   - `API_MODEL`

#### 步骤 3: 修改工作流文件

在 `.github/workflows/ci-api-optimized.yml` 中，取消注释 `environment` 行：

```yaml
jobs:
  test-api-optimized:
    runs-on: ubuntu-latest
    environment: production  # 取消注释，使用您创建的环境名称
    # ...
```

#### Environment 名称注意事项

- **命名规范**: 使用小写字母和连字符（如 `production`、`staging`、`development`）
- **唯一性**: 每个环境名称在同一仓库中必须唯一
- **描述性**: 选择能够清晰描述该环境用途的名称
- **工作流引用**: 工作流文件中的 `environment` 名称必须与创建的环境名称完全一致（区分大小写）

#### Repository Secrets vs Environment Secrets

| 特性 | Repository Secrets | Environment Secrets |
|------|-------------------|---------------------|
| 配置位置 | Settings → Secrets | Settings → Environments → Environment secrets |
| 工作流配置 | 无需额外配置 | 需要在 job 中添加 `environment: <name>` |
| 权限控制 | 仓库级别 | 环境级别（可设置审批者） |
| 适用场景 | 简单项目 | 多环境、需要权限控制 |

### 工作流文件

已配置的工作流文件：
- `.github/workflows/ci.yml` - 测试 `rel.py`（使用 HanLP）
- `.github/workflows/ci-api-optimized.yml` - 测试 `rel_api_optimized.py`（使用 LLM API）

### 工作流说明

**ci.yml**:
- 自动安装 Java 和 HanLP 依赖
- 缓存 HanLP 数据文件
- 运行 `rel.py` 进行测试

**ci-api-optimized.yml**:
- 自动安装依赖
- 如果设置了 `API_KEY`，运行 `rel_api_optimized.py`
- 如果未设置 `API_KEY`，只进行语法检查

### 触发条件

工作流会在以下情况自动运行：
- 推送到 `main` 或 `master` 分支
- 创建 Pull Request
- 手动触发（Workflow dispatch）

## 环境变量

### 本地运行

```bash
# LLM API 配置
export API_BASE_URL="https://miaodi.zeabur.app"
export API_KEY="your-api-key"
export API_MODEL="deepseek-ai/DeepSeek-V3-0324"

# 或使用命令行参数
python rel_api_optimized.py \
    --book 冬日重现 \
    --base_url https://miaodi.zeabur.app \
    --api_key your-api-key \
    --model deepseek-ai/DeepSeek-V3-0324
```

### GitHub Actions

- **Repository Secrets**: 环境变量会自动从 Secrets 读取，无需手动设置
- **Environment Secrets**: 需要在工作流文件中指定 `environment` 名称

## 工作流程

### rel.py（HanLP 共现统计）

```
1. 使用 HanLP 提取所有人名
2. 统计人名共现关系
3. 过滤和去重
4. 生成关系图和 Excel
```

### rel_api_optimized.py（LLM API 分析）

```
1. 使用 HanLP 提取高频人名（filter_nr 过滤）
2. 找出包含至少两个人名的句子（最多200个）
3. 提取句子所在的段落上下文
4. 分批交给 LLM 分析人物关系
5. 生成关系图和 Excel
```

## 依赖说明

### 必需依赖

- `tqdm` - 进度条
- `numpy` - 数值计算
- `networkx` - 图分析
- `matplotlib` - 绘图
- `pandas` - 数据处理
- `openpyxl` - Excel 导出
- `requests` - HTTP 请求

### 可选依赖

- `pyhanlp` + `JPype1` - 人名识别（`rel.py` 和 `rel_api_optimized.py` 需要）
- `paddlenlp` - UIE 模型（`rel_uie.py` 需要，可选）

## 注意事项

1. **API 密钥安全**: 不要将 API 密钥提交到代码仓库，使用 GitHub Secrets
2. **HanLP 数据**: 首次运行会自动下载 HanLP 数据文件（约 600MB）
3. **API 限流**: `rel_api_optimized.py` 会自动处理 API 限流（每批之间延迟 0.5 秒）
4. **Token 消耗**: `rel_api_optimized.py` 只分析关键段落，大幅节省 token
5. **Environment 名称**: 如果使用 Environment Secrets，工作流文件中的 `environment` 名称必须与创建的环境名称完全一致

## 故障排查

### GitHub Secrets 相关问题

- **Secrets 未生效**: 检查工作流文件中是否正确引用了 `secrets.XXX`
- **Environment secrets 未生效**: 检查是否在工作流文件中添加了 `environment: <name>`
- **环境名称不匹配**: 确保工作流文件中的环境名称与创建的环境名称完全一致（区分大小写）

### HanLP 相关问题

- **Java 未安装**: 需要 Java 8 或更高版本
- **数据文件缺失**: 首次运行会自动下载，或从缓存恢复

### API 相关问题

- **连接失败**: 检查 `API_BASE_URL` 是否正确
- **认证失败**: 检查 `API_KEY` 是否正确
- **模型不存在**: 检查 `API_MODEL` 是否正确

## 许可证

MIT License

