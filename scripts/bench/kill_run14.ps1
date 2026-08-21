# Hard-stop for the run14 benchmark: kills wrapper, driver, and daemon.
# Scheduled via schtasks so the cutoff fires even if the supervising agent
# session is dead. Grading (WSL Docker) is not touched.
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'run_shards_sequential|swe_driver\.py|from atn\.cli import main'
}
foreach ($p in $targets) {
    try { Stop-Process -Id $p.ProcessId -Force -Confirm:$false } catch {}
}
"$(Get-Date -Format o) killed $($targets.Count) processes" |
    Out-File -Append -Encoding utf8 C:\code\autonet\scripts\bench\results\run14_hardstop.log
