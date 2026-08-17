# Auto-push toutes les 60 secondes vers GitHub
# Usage: powershell -ExecutionPolicy Bypass -File .\programme\auto-push-minute.ps1

$repoPath = "c:\Users\MarcialDomche\Desktop\Lundi_Reunion\Ki_Projekt\gaushorn_shot_classifier"
$gitExe = "C:\Program Files\Git\bin\git.exe"
$intervalSeconds = 60

Set-Location $repoPath

Write-Host "Surveillance active : push automatique toutes les $intervalSeconds secondes"
Write-Host "Appuyez sur Ctrl+C pour arreter"

while ($true) {
    $status = & $gitExe status --short

    if ($status) {
        Write-Host ""
        Write-Host "Modifications detectees :"
        $status

        & $gitExe add -A

        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $message = "Auto-commit: synced at $timestamp"
        & $gitExe commit -m $message

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Commit vide ou echec du commit. Nouvelle tentative dans $intervalSeconds s..."
            Start-Sleep -Seconds $intervalSeconds
            continue
        }

        $branch = (& $gitExe rev-parse --abbrev-ref HEAD).Trim()
        Write-Host "Push vers GitHub (branche: $branch)"
        & $gitExe push -u origin $branch

        if ($LASTEXITCODE -eq 0) {
            Write-Host "Synchronisation reussie !"
        } else {
            Write-Host "Echec du push. Verifie le depot GitHub et le token."
        }
    }

    Start-Sleep -Seconds $intervalSeconds
}
