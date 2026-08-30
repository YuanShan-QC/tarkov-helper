# ============================================================
#  塔科夫助手 - 等待后端就绪脚本
#  轮询 /api/health，后端就绪即退出（exit 0）；超时退出 1。
#  由 start.bat 调用：保证浏览器在服务真正可用后才打开页面。
# ============================================================
$Url = 'http://127.0.0.1:8100/api/health'
$MaxSeconds = 180   # 最多等 3 分钟（首次运行联网下载数据可能较久）
$Interval = 2       # 每 2 秒检查一次
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$ok = $false

while ($sw.Elapsed.TotalSeconds -lt $MaxSeconds) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch {
        # 后端未就绪，继续等
    }
    Start-Sleep -Seconds $Interval
    $el = [int]$sw.Elapsed.TotalSeconds
    if (($el -gt 0) -and (($el / $Interval) -eq [math]::Floor($el / $Interval))) {
        Write-Host ("  等待后端启动... {0} 秒" -f $el)
    }
}

if ($ok) {
    Write-Host '  后端服务已就绪，正在打开页面...'
    exit 0
} else {
    Write-Host "  [警告] 等待超时（${MaxSeconds} 秒），后端仍未就绪，仍将尝试打开页面"
    exit 1
}
