# 把工作区各课程的笔记以「文件级符号链接」挂进 Obsidian 仓库
# 用法：以管理员身份运行 PowerShell，然后执行本文件
#   Start-Process pwsh -Verb RunAs -ArgumentList '-NoProfile','-File','<本文件路径>'
# 幂等：已存在的链接会跳过，可反复运行
$ws   = 'E:\SUFE\junior\term1'
$repo = 'D:\ObsidianRepository\Machine-Learning'
$map  = @{
  '高维数据分析' = '高维数据分析'
  '宏观经济学'   = '宏观'
  '货币银行学'   = '货币'
  '机器学习'     = '机器学习'
  '随机过程'     = '随机过程'
  '微观经济学'   = '微观'
}
foreach ($c in $map.Keys) {
  # 课程笔记目录：兼容 '01_笔记' 与 '笔记' 两种整理方式
  $src = $null
  foreach ($sub in @('01_笔记', '笔记')) {
    $p = Join-Path $ws "$c\$sub"
    if (Test-Path $p) { $src = $p; break }
  }
  if (-not $src) { continue }

  $dstRoot = Join-Path $repo "$($map[$c])\笔记"
  if (-not (Test-Path $dstRoot)) { New-Item -ItemType Directory $dstRoot -Force | Out-Null }

  Get-ChildItem $src -Recurse -File -Filter *.md | ForEach-Object {
    $rel = $_.FullName.Substring($src.Length + 1)
    $dst = Join-Path $dstRoot $rel
    if (Test-Path -LiteralPath $dst) { return }        # 已有链接，跳过
    $parent = Split-Path $dst -Parent
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory $parent -Force | Out-Null }
    New-Item -ItemType SymbolicLink -Path $dst -Target $_.FullName | Out-Null
    Write-Host "链接: $c\$rel"
  }
}

Write-Host ''
Write-Host '=== 断链自检 ==='
Get-ChildItem $repo -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint -and -not (Test-Path -LiteralPath $_.Target) } |
  ForEach-Object { Write-Host "断链: $($_.FullName)" }
Write-Host '完成（上方无「断链」即正常）'
