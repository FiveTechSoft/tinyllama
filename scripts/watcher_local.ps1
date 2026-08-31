# watcher_local.ps1 - autocommit del checkpoint local al repo
# corre en background en el PC; cada 10 min sube checkpoint.chk/meta si cambio
#
# instalacion (una vez): crear tarea programada o ejecutar al inicio
#   powershell -ExecutionPolicy Bypass -File watcher_local.ps1

Repo = $null  # placeholder para evitar parser quirks

$repo = "C:\tinyllama"
$lastWrite = [datetime]::MinValue

while ($true) {
    Start-Sleep -Seconds 600   # cada 10 min
    try {
        $chk = Join-Path $repo "checkpoint.chk"
        if (-not (Test-Path $chk)) { continue }

        $info = Get-Item $chk
        $age = (Get-Date) - $info.LastWriteTime
        if ($age.TotalMinutes -lt 11) {   # recien escrito -> entrenamiento activo
            Push-Location $repo
            git add adaptive/checkpoint.chk adaptive/checkpoint.meta 2>$null
            git add checkpoint.chk checkpoint.meta 2>$null
            $st = git status --porcelain | Select-String "checkpoint"
            if ($st) {
                git commit -m "checkpoint local: entrenamiento en vivo $(Get-Date -Format 'MM-dd HH:mm')" 2>$null
                git pull --rebase origin main 2>$null | Out-Null
                git push 2>$null
                Write-Output ("subido checkpoint " + (Get-Date -Format "HH:mm:ss"))
            }
            if (Test-Path .git\REBASE_HEAD) { git rebase --abort 2>$null }
            Pop-Location
        }
    } catch {
        Write-Output ("watcher err: " + $_.Message)
    }
}