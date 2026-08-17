# Script PowerShell : Auto-Commit et Push vers GitHub
# Description:
#   Cette version surveille les changements dans le dépôt local.
#   Quand un fichier est modifié, elle l'ajoute, crée un commit automatique,
#   puis pousse vers GitHub.
#
# Utilisation:
#   .\auto-push.ps1
#
# Remarque:
#   Ce script est une version simple et plus courte, utile pour les tests rapides.

# Chemin du dépôt Git local
$repoPath = "c:\Users\MarcialDomche\Desktop\Lundi_Reunion\Ki_Projekt\gaushorn_shot_classifier"

# Se placer dans le dossier du projet
Set-Location $repoPath

# Message de départ
Write-Host "👀 Surveillance active : on attend les changements..." -ForegroundColor Cyan
Write-Host "Appuyez sur Ctrl+C pour arreter." -ForegroundColor Yellow

# Boucle infinie de surveillance
while ($true) {
    # Vérifie si le dépôt contient des modifications non suivies ou modifiées
    $status = git status --short

    if ($status) {
        Write-Host "`n📝 Changements detectes :" -ForegroundColor Green
        Write-Host $status

        # Ajoute tous les fichiers modifiés dans l'index Git
        git add -A

        # Crée un message de commit avec l'heure actuelle
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $message = "Auto-commit: Changes synchronized at $timestamp"

        # Réalise le commit
        git commit -m $message

        # Envoie le commit vers GitHub
        Write-Host "🚀 Push vers GitHub..." -ForegroundColor Cyan
        git push -u origin main

        # Vérifie le résultat du push
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Synchronise avec GitHub !" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Erreur lors du push. Verifie votre accès GitHub." -ForegroundColor Yellow
        }
    }

    # Pause avant la prochaine vérification
    Start-Sleep -Seconds 10
}
