# PaddleNLP UIE 部署指南

## 关于本地推理

**是的，PaddleNLP UIE 需要本地推理。**

UIE 模型会在首次运行时自动下载到本地，然后进行推理。模型文件通常存储在：
- Linux/Mac: `~/.paddlenlp/models/` 或 `~/.paddlenlp/taskflow/`
- Windows: `C:\Users\<用户名>\.paddlenlp\models\` 或 `C:\Users\<用户名>\.paddlenlp\taskflow\`

### 模型大小对比

| 模型 | 参数量 | 文件大小 | 适用场景 |
|------|--------|----------|----------|
| `uie-nano` | 最小 | ~50MB | CI/CD、资源受限环境 |
| `uie-tiny` | 小 | ~100MB | 轻量级部署 |
| `uie-mini` | 中 | ~200MB | 平衡性能和资源 |
| `uie-base` | 大 | ~400MB | 默认选择，性能较好 |
| `uie-medium` | 更大 | ~800MB | 高性能需求 |
| `uie-large` | 最大 | ~1.5GB | 最佳性能 |

## GitHub Actions 部署

### 挑战

在 GitHub Actions 中部署 UIE 会遇到以下问题：

1. **模型下载时间长**
   - 首次运行需要下载模型（几百MB到几GB）
   - GitHub Actions 有超时限制（默认 6 小时，但单个 job 可能更短）

2. **资源限制**
   - 免费版：2 核 CPU，7GB RAM，14GB 磁盘
   - 无 GPU 支持（只能 CPU 推理）

3. **网络问题**
   - 模型下载可能失败或超时
   - 需要配置镜像源（国内用户）

### 优化方案

#### 方案 1: 使用轻量级模型（推荐）

在 CI 环境中使用 `uie-tiny` 或 `uie-nano`：

```yaml
env:
  UIE_MODEL: uie-tiny  # 或 uie-nano
```

#### 方案 2: 缓存模型文件

在 CI 配置中缓存模型：

```yaml
- name: Cache UIE model
  uses: actions/cache@v4
  with:
    path: ~/.paddlenlp
    key: ${{ runner.os }}-uie-model-${{ env.UIE_MODEL }}-v1
    restore-keys: |
      ${{ runner.os }}-uie-model-
```

#### 方案 3: 使用预下载脚本

提前下载模型到缓存中。

## CI 配置示例

### 完整 CI 配置（推荐）

```yaml
name: CI with UIE

on:
  push:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  test-uie:
    runs-on: ubuntu-latest
    timeout-minutes: 60  # 设置超时时间
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.9"
    
    - name: Cache pip packages
      uses: actions/cache@v4
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    
    - name: Cache UIE model
      uses: actions/cache@v4
      with:
        path: ~/.paddlenlp
        key: ${{ runner.os }}-uie-tiny-model-v1
        restore-keys: |
          ${{ runner.os }}-uie-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install paddlenlp>=2.5.0 pandas>=1.3.0 openpyxl>=3.0.0
        pip install tqdm numpy networkx matplotlib
    
    - name: Install Chinese fonts
      run: |
        sudo apt-get update
        sudo apt-get install -y fonts-wqy-zenhei fonts-wqy-microhei
    
    - name: Test UIE extraction
      env:
        CI: true
        UIE_MODEL: uie-tiny  # 在 CI 中使用轻量级模型
        MPLBACKEND: Agg
      run: |
        mkdir -p output
        # 使用小样本测试，避免超时
        python rel_uie.py --book 冬日重现 --batch_size 16 --max_length 256 || {
          echo "⚠️ UIE 测试失败，可能是模型下载超时或资源不足"
          exit 0  # 在 CI 中允许失败，避免阻塞主流程
        }
      continue-on-error: true  # 允许失败，因为模型下载可能超时
    
    - name: Upload results
      if: always()
      uses: actions/upload-artifact@v4
      with:
        name: uie-results
        path: output/
        if-no-files-found: warn
```

### 最小化 CI 配置（仅测试）

如果只想测试代码语法，不运行完整推理：

```yaml
- name: Test UIE script syntax
  run: |
    python -m py_compile rel_uie.py
```

## 环境变量配置

可以通过环境变量控制 UIE 行为：

```bash
# 选择模型（在 CI 中使用较小的模型）
export UIE_MODEL=uie-tiny

# 设置模型下载镜像（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 PaddlePaddle 后端
export FLAGS_selected_gpus=  # 空值表示使用 CPU
```

## 备选方案

### 方案 A: 仅在本地运行 UIE

在 CI 中只测试 `rel.py`（HanLP），UIE 版本仅在本地使用：

```yaml
- name: Test original rel.py
  run: python rel.py --book 冬日重现

- name: Skip UIE test in CI
  run: echo "UIE 测试在本地进行，CI 中跳过"
```

### 方案 B: 使用 GitHub Actions 的 Self-hosted Runner

如果有自己的服务器，可以配置 self-hosted runner，获得更多资源和 GPU 支持。

### 方案 C: 使用模型服务 API

如果不想在 CI 中下载模型，可以考虑：
- 使用 PaddleNLP 的在线 API（如果有）
- 自建模型服务
- 使用其他轻量级 NER 方案

## 推荐配置

对于 GitHub Actions，推荐配置：

1. **模型选择**: `uie-tiny`（在 CI 中）
2. **缓存策略**: 缓存 `~/.paddlenlp` 目录
3. **超时设置**: 设置合理的超时时间（30-60 分钟）
4. **错误处理**: 使用 `continue-on-error: true` 允许失败
5. **分批处理**: 减小 `batch_size` 和 `max_length` 参数

## 示例：优化的 rel_uie.py 调用

```bash
# 本地运行（使用完整模型）
python rel_uie.py --book 冬日重现

# CI 环境运行（自动使用轻量级模型）
CI=true UIE_MODEL=uie-tiny python rel_uie.py --book 冬日重现 --batch_size 16 --max_length 256
```

## 故障排查

### 问题 1: 模型下载超时

**解决方案**:
- 使用 `uie-tiny` 或 `uie-nano`
- 增加 CI 超时时间
- 使用缓存机制

### 问题 2: 内存不足

**解决方案**:
- 减小 `batch_size`（如从 32 降到 16 或 8）
- 减小 `max_length`（如从 512 降到 256）
- 使用更小的模型

### 问题 3: 推理速度慢

**解决方案**:
- 在 CI 中只测试小样本
- 使用 `continue-on-error: true` 允许超时
- 考虑只在本地运行 UIE 测试

## 总结

- ✅ **本地推理**: UIE 需要本地推理，模型会自动下载
- ⚠️ **CI 部署**: 可以部署，但需要优化（使用小模型、缓存、允许失败）
- 💡 **推荐**: 在 CI 中使用 `uie-tiny` 或 `uie-nano`，并启用缓存

