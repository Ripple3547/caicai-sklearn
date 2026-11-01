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
  $src = Join-Path $ws "$c\笔记"
  if (-not (Test-Path $src)) { continue }
  $dstRoot = Join-Path $repo "$($map[$c])\笔记"
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