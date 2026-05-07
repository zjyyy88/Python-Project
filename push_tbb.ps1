#!/usr/bin/env pwsh
# 推送tbb-LPSCI文件夹到云端

Set-Location "c:\Users\ZHANGJY02\PycharmProjects\PythonProject"

Write-Host "=== Git 状态检查 ===" -ForegroundColor Green
git status

Write-Host "`n=== 检查tbb-LPSCI文件夹 ===" -ForegroundColor Green
if (Test-Path "tbb-LPSCI") {
    Write-Host "✓ tbb-LPSCI 文件夹存在" -ForegroundColor Green
    $itemCount = (Get-ChildItem -Path "tbb-LPSCI" -Recurse | Measure-Object).Count
    Write-Host "  包含 $itemCount 个文件/文件夹" -ForegroundColor Cyan
} else {
    Write-Host "✗ tbb-LPSCI 文件夹不存在" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 添加tbb-LPSCI到git ===" -ForegroundColor Green
git add tbb-LPSCI/

Write-Host "`n=== 检查待提交更改 ===" -ForegroundColor Green
git status

Write-Host "`n=== 提交更改 ===" -ForegroundColor Green
$commitMsg = "Sync: Add/Update tbb-LPSCI folder - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
git commit -m $commitMsg

Write-Host "`n=== 推送到远程仓库 ===" -ForegroundColor Green
git push

Write-Host "`n✓ 推送完成!" -ForegroundColor Green
git log --oneline -1
