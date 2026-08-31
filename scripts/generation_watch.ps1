# sample_watch.ps1 - genera 1样本 por hora (sin molestar al train) y loguea
# la evolucion del texto generadodel modelo adaptativo
$repo = "C:\tinyllama"
$log = "$repo\adaptive\generation_log.txt"

while ($true) {
    Start-Sleep -Seconds 3600   # cada hora
    try {
        $chk = Join-Path $repo "checkpoint.chk"
        if (-not (Test-Path $chk)) { continue }
        $age = ((Get-Date) - (Get-Item $chk).LastWriteTime).TotalMinutes
        if ($age -gt 12) { continue }   # entrenamiento muerto -> no samplear

        Push-Location $repo
        $line = & ".\adaptive\bpe_gen.exe" "..\checkpoint.chk" ".\adaptive\vocab.bpe" "Give three tips for staying healthy" 60 0.9 60 2>$null | Select-Object -Last 1
        Pop-Location
        # decodificar a texto con python y logear
        python -c "
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
lines = open(r'C:\tinyllama\adaptive\vocab.bpe', encoding='utf-8').read().splitlines()
id2s = {}
for l in lines[1:]:
    p = l.split(chr(9))
    if len(p) >= 3: id2s[int(p[0])] = p[1]+p[2].replace(chr(13),'')
ids = '''$line'''.split()
txt = ''.join(id2s.get(int(v), chr(v)) for v in ids if int(v) > 3)
import subprocess, datetime
with open(r'$repo\adaptive\generation_log.txt', 'a', encoding='utf-8') as f:
    f.write(f'{txt}\n')
print('logueado:', txt[:80] if False else txt[:0])
" 2>$null
        Add-Content $log ("---- " + (Get-Date -Format "MM-dd HH:mm") + " (step actual en logs) ----")
    } catch { Write-Output ("err: " + $_.Message) }
}