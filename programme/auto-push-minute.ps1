# Auto-push toutes les 60 secondes vers GitHub
# Usage: powershell -ExecutionPolicy Bypass -File .\programme\auto-push-minute.ps1

$repoPath = "c:\Users\MarcialDomche\Desktop\Lundi_Reunion\Ki_Projekt\gaushorn_shot_classifier"
$gitExe = "C:\Program Files\Git\bin\git.exe"
$intervalSeconds = 60
$logFile = Join-Path $env:TEMP "gaushorn_auto_push.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Set-Location $repoPath

Write-Log "Demarrage du script d'auto-push. Intervalle: $intervalSeconds secondes"
Write-Log "Dossier de travail: $repoPath"
Write-Log "Fichier de log: $logFile"

while ($true) {
    $status = & $gitExe status --short

    if ($status) {
        Write-Log "Modifications detectees."
        $status | ForEach-Object { Write-Log $_ }

        & $gitExe add -A

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $message = "Auto-commit: synced at $timestamp"
        $commitOutput = & $gitExe commit -m $message 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-Log "Commit vide ou echec du commit. Nouvelle tentative dans $intervalSeconds secondes."
            if ($commitOutput) {
                $commitOutput | ForEach-Object { Write-Log $_ }
            }
            Start-Sleep -Seconds $intervalSeconds
            continue
        }

        $branch = (& $gitExe rev-parse --abbrev-ref HEAD).Trim()
        Write-Log "Push vers GitHub (branche: $branch)"
        $pushOutput = & $gitExe push -u origin $branch 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Log "Synchronisation reussie !"
            if ($pushOutput) {
                $pushOutput | ForEach-Object { Write-Log $_ }
            }
        } else {
            Write-Log "Echec du push. Verifie le depot GitHub et le token."
            if ($pushOutput) {
                $pushOutput | ForEach-Object { Write-Log $_ }
            }
        }
    }

    Start-Sleep -Seconds $intervalSeconds
}
