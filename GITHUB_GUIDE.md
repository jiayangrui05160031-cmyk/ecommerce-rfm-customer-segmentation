# GitHub 部署操作指南

## 一、5 分钟推送到 GitHub

### 1. 前置条件
- 安装 Git：https://git-scm.com/download/win
- 注册 GitHub 账号：https://github.com
- （推荐）配置 SSH key：https://docs.github.com/zh/authentication/connecting-to-github-with-ssh

### 2. **重要：先删掉大文件，否则 push 失败**

GitHub 单文件不能超过 25MB，本项目的大文件**不要提交**：

```powershell
cd D:\cluade_outpute\ecommerce-rfm-customer-segmentation

# 1. 原始 xlsx（43MB）—— 已拆成 5 个 CSV 分块，删原文件
Remove-Item data\raw\online_retail_II.xlsx

# 2. 清洗后的 pkl（51MB）—— 每次运行 01 notebook 会自动重建
Remove-Item data\processed\cleaned_retail.pkl

# 3. .gitignore 已经帮你过滤这些文件，以后再生成也不会被误提交
```

可以 `git status` 确认没有大文件后再继续。

### 3. 在 GitHub 创建空仓库
1. 登录 GitHub → 右上角 **+** → **New repository**
2. Repository name 填：`ecommerce-rfm-customer-segmentation`
3. Description 填：`基于 RFM 模型与 K-Means 聚类的电商用户分群与精准营销分析`
4. 选 **Public**（公开，方便面试官看）
5. **不要**勾选 Add a README / Add .gitignore / Choose a license
6. 点 **Create repository**

### 4. 在项目目录里执行推送脚本
```powershell
cd D:\cluade_outpute\ecommerce-rfm-customer-segmentation
.\push_to_github.ps1
```
按提示输入 GitHub 用户名即可。

### 5. 如果推送失败（认证问题）
最常见原因：用了 HTTPS 但没配 PAT。两种解法：
- **方案 A（推荐）**：配 SSH key，然后改 remote 为 SSH 格式：
  ```powershell
  git remote set-url origin git@github.com:你的用户名/ecommerce-rfm-customer-segmentation.git
  git push -u origin main
  ```
- **方案 B**：用 Personal Access Token 推送：
  https://github.com/settings/tokens → 生成 token → 推送时把密码换成 token

## 二、数据集拆分说明

### 为什么拆分？
原始 Online Retail II 数据集 ~43MB，cleaned_retail.pkl ~51MB，
都超过 GitHub 单文件 25MB 上限。

### 怎么拆的？
`scripts/split_data.py` 按行切分 xlsx，输出 5 个 < 20MB 的 CSV 分块。
你可以重新运行该脚本自定义切分方式（修改 `TARGET_CHUNK_MB`）。

### clone 下来后怎么用？
`src/data_loader.py` 自动判断：
- 如果 `data/raw/online_retail_II.xlsx` 存在 → 直接读
- 否则 → 拼接 `data/raw/online_retail_II_part*.csv` 分块

所以克隆项目后，**第一次运行**会自动从 5 个 CSV 拼出完整数据，
之后 `data/processed/cleaned_retail.pkl` 会被缓存（.gitignore 已过滤，不会被误传）。

## 二、让项目更好看（5 个加分动作）

### 1. 完善 About 区
在 GitHub 仓库页面点 **⚙️ Settings → General**：
- Description：`基于 RFM 模型与 K-Means 聚类的电商用户分群与精准营销分析`
- Website：可填你的个人博客或 LinkedIn
- Topics：`python`, `pandas`, `scikit-learn`, `data-analysis`, `customer-segmentation`, `rfm`, `kmeans`, `marketing-analytics`, `business-analytics`

### 2. 把可视化图上传到 images/
跑完 4 个 notebook 后：
- `images/rfm_3d.html` 直接保留
- `images/cluster_2d.png` / `rfm_radar.png` / `revenue_pareto.png` / `conversion_funnel.png` 上传
- 替换 README 里的图片占位

### 3. 在 README 顶部加一张 banner
可以用 [Shields.io](https://shields.io) 生成项目徽章。

### 4. 加 GitHub Actions（可选）
`.github/workflows/test.yml`：
```yaml
name: test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python src/verify_pipeline.py
```

### 5. Releases & Tags
项目稳定后，在 GitHub 上发 v1.0.0 release，HR 一眼看到版本号会觉得更专业。

## 三、常见问题

| 问题 | 解决 |
|------|------|
| 推送时文件名含中文报错 | 把项目挪到英文路径，如 `D:\projects\ecommerce-rfm` |
| notebook 太大推不上去 | 用 `git lfs` 跟踪大文件，或用 nbstripout 去掉 output |
| Excel 文件打不开 | 安装 `openpyxl`：pip install openpyxl |
| KMeans 报警告 ConvergenceWarning | 调高 n_init 或加 max_iter |

## 四、推荐项目名（备选）

如果 `ecommerce-rfm-customer-segmentation` 太长，可以考虑：
- `rfm-customer-segmentation`
- `ecommerce-customer-analytics`
- `retail-rfm-kmeans`
- `customer-value-segmentation`
