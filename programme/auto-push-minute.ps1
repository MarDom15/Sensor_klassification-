# Script: auto-push-minute.ps1
# Description:
#   Surveille le dépôt Git local toutes les 60 secondes.
#   Si des fichiers ont changé, il les ajoute, crée un commit automatique,
#   puis pousse vers GitHub sur la branche active.
#
# Utilisation:
#   powershell -ExecutionPolicy Bypass -File .\programme\auto-push-minute.ps1
#
# Remarque:
#   Le journal est écrit dans le dossier temporaire Windows pour éviter
#   que le script ne se committe lui-même en boucle.

# Chemin du dépôt Git local à surveiller
$repoPath = "c:\Users\MarcialDomche\Desktop\Lundi_Reunion\Ki_Projekt\gaushorn_shot_classifier"

# Chemin vers l'exécutable Git installé sur Windows
$gitExe = "C:\Program Files\Git\bin\git.exe"

# Intervalle de vérification en secondes
$intervalSeconds = 60

# Fichier de log externe au dépôt pour éviter les boucles de commit
$logFile = Join-Path $env:TEMP "gaushorn_auto_push.log"

# Fonction utilitaire pour écrire un message dans la console et dans le fichier log, super interessant ce truc , salöut moi ca va 
function Write-Log {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"

    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

# Se placer dans le dossier du projet
Set-Location $repoPath

# Message d'initialisation
Write-Log "Demarrage du script d'auto-push. Intervalle: $intervalSeconds secondes"
Write-Log "Dossier de travail: $repoPath"
Write-Log "Fichier de log: $logFile"

# Boucle principale de surveillance
while ($true) {
    # Vérifie l'état Git du dépôt (fichiers modifiés, nouveaux, supprimés)
    $status = & $gitExe status --short

    # Si le dépôt contient des changements
    if ($status) {
        Write-Log "Modifications detectees."
        $status | ForEach-Object { Write-Log $_ }

        # Ajout de tous les changements dans l'index Git
        & $gitExe add -A

        # Message de commit basé sur l'heure actuelle
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $message = "Auto-commit: synced at $timestamp"

        # Crée un commit ; on capture la sortie pour gérer les cas sans modification
        $commitOutput = & $gitExe commit -m $message 2>&1

        # Si le commit a échoué (par exemple: rien à committer)
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Commit vide ou echec du commit. Nouvelle tentative dans $intervalSeconds secondes."
            if ($commitOutput) {
                $commitOutput | ForEach-Object { Write-Log $_ }
            }
            Start-Sleep -Seconds $intervalSeconds
            continue
        }

        # Détecte la branche actuelle pour pousser dessus
        $branch = (& $gitExe rev-parse --abbrev-ref HEAD).Trim()
        Write-Log "Push vers GitHub (branche: $branch)"

        # Envoi vers le dépôt distant
        $pushOutput = & $gitExe push -u origin $branch 2>&1

        # Vérifie si le push a réussi
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

    # Pause avant la prochaine vérification
    Start-Sleep -Seconds $intervalSeconds
}
