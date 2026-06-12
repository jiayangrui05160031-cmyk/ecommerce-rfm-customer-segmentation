<#
.SYNOPSIS
    一键推送电商 RFM 项目到 GitHub。

.DESCRIPTION
    假设：
    1. 你已经安装 Git (https://git-scm.com/download/win)
    2. 你已经在 GitHub 上创建好空仓库（不要勾选 README）
    3. 你的 SSH key 或 Personal Access Token 已配好

.NOTES
    使用方法（PowerShell）：
        1. 打开 PowerShell，进入项目目录
        2. .\push_to_github.ps1
        3. 按提示输入 GitHub 用户名和仓库名
#>

param(
    [string]$GitHubUser = "",
    [string]$RepoName = "ecommerce-rfm-customer-segmentation"
)

# ===== 颜色输出 =====
function Write-Step($msg)    { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)      { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)     { Write-Host "[X] $msg" -ForegroundColor Red }

# ===== 检查 git =====
Write-Step "检查 Git 环境"
$gitVersion = (& git --version) 2>$null
if (-not $gitVersion) {
    Write-Err "未检测到 Git，请先安装：https://git-scm.com/download/win"
    exit 1
}
Write-Ok "Git 已安装：$gitVersion"

# ===== 收集 GitHub 信息 =====
if (-not $GitHubUser) {
    $GitHubUser = Read-Host "请输入你的 GitHub 用户名"
}
Write-Ok "用户名：$GitHubUser"
Write-Ok "仓库名：$RepoName"

# ===== 切换到脚本所在目录 =====
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
Write-Ok "项目目录：$projectDir"

# ===== git init =====
if (-not (Test-Path ".git")) {
    Write-Step "初始化 Git 仓库"
    & git init
    & git branch -M main
    Write-Ok "已初始化 main 分支"
} else {
    Write-Warn ".git 已存在，跳过初始化"
}

# ===== 配置 user（如果未配置） =====
$userName = (& git config user.name) 2>$null
$userEmail = (& git config user.email) 2>$null
if (-not $userName) {
    $name = Read-Host "请输入 Git 用户名（用于 commit）"
    & git config user.name $name
}
if (-not $userEmail) {
    $email = Read-Host "请输入 Git 邮箱（用于 commit）"
    & git config user.email $email
}
Write-Ok "Git user 配置完成"

# ===== 写入 .gitignore =====
Write-Step "写入 .gitignore（避免上传数据和图片）"
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*.so
.venv/
venv/

# Jupyter
.ipynb_checkpoints/
profile_default/

# Data & images (太大，可选择上传)
data/raw/
data/processed/*.pkl
images/*.png
images/*.html
!images/.gitkeep

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"@
Set-Content -Path ".gitignore" -Value $gitignore -Encoding utf8
Write-Ok ".gitignore 已写入"

# 保留 images 空目录
if (-not (Test-Path "images")) { New-Item -ItemType Directory -Path "images" | Out-Null }
if (-not (Test-Path "images/.gitkeep")) { New-Item -ItemType File -Path "images/.gitkeep" | Out-Null }

# ===== 添加并提交 =====
Write-Step "添加文件并提交"
& git add .
& git status --short
& git commit -m "init: RFM customer segmentation project (data + notebooks + src)"

# ===== 配置 remote 并推送 =====
$remoteUrl = "https://github.com/$GitHubUser/$RepoName.git"
Write-Step "配置 remote: $remoteUrl"
& git remote remove origin 2>$null
& git remote add origin $remoteUrl

Write-Step "推送到 GitHub"
& git push -u origin main
if ($?) {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "  🎉 推送成功！" -ForegroundColor Green
    Write-Host "  仓库地址：https://github.com/$GitHubUser/$RepoName" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
} else {
    Write-Err "推送失败，可能是认证问题。"
    Write-Host "请检查：" -ForegroundColor Yellow
    Write-Host "  1. GitHub 用户名是否正确" -ForegroundColor Yellow
    Write-Host "  2. 仓库是否已创建" -ForegroundColor Yellow
    Write-Host "  3. 是否已配置 SSH key 或 Personal Access Token" -ForegroundColor Yellow
    Write-Host "  4. 网络是否能访问 github.com" -ForegroundColor Yellow
}
