# Script PowerShell : Auto-Commit et Push tous les changements
# Utilisation : .\auto-push.ps1

$repoPath = "c:\Users\MarcialDomche\Desktop\Lundi_Reunion\Ki_Projekt\gaushorn_shot_classifier"
Set-Location $repoPath

Write-Host "👀 Watching for changes..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow

while ($true) {
    # Vérifier s'il y a des changements
    $status = git status --short
    
    if ($status) {
        Write-Host "`n📝 Changements détectés :" -ForegroundColor Green
        Write-Host $status
        
        # Ajouter tous les fichiers
        git add -A
        
        # Créer un commit avec timestamp
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $message = "Auto-commit: Changes synchronized at $timestamp"
        git commit -m $message
        
        # Push vers GitHub
        Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
        git push -u origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Synchronisé avec GitHub!" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Erreur lors du push" -ForegroundColor Yellow
        }
    }
    
    # Vérifier tous les 10 secondes
    Start-Sleep -Seconds 10
}
