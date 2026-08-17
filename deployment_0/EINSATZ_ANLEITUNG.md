# Einsatz auf dem Raspberry Pi — Schritt für Schritt

Diese Anleitung zeigt, wie man das fertige Modell (`models/shot_vs_parasite_svm.joblib`)
wirklich auf einem Raspberry Pi zum Laufen bringt.

## 1. Was man braucht

- Ein Raspberry Pi (jedes Modell reicht — sogar ein Pi Zero 2 W. Das Modell
  ist winzig und braucht kaum Rechenleistung).
- Eine microSD-Karte (mind. 8 GB) und ein Kartenlesegerät.
- Raspberry Pi OS (früher "Raspbian"), am besten die **64-Bit Lite-Version**
  (ohne Desktop, spart Platz und startet schneller).
- Eine Internetverbindung für den Pi (WLAN oder Kabel), zumindest für die
  einmalige Installation der Pakete.
- Den Ordner `gaushorn_shot_classifier/` (dieses Projekt).

## 2. Raspberry Pi OS vorbereiten

1. **Raspberry Pi Imager** auf dem eigenen Computer installieren
   (imager.raspberrypi.com).
2. SD-Karte einlegen, Imager öffnen, "Raspberry Pi OS Lite (64-bit)" wählen.
3. Im Imager auf das Zahnrad-Symbol klicken (erweiterte Einstellungen) und:
   - Hostname vergeben (z. B. `gaushorn-pi`)
   - SSH aktivieren, Benutzername + Passwort setzen
   - WLAN-Zugangsdaten eintragen (falls kein Kabel genutzt wird)
4. Karte beschreiben lassen, in den Pi einlegen, Pi starten.
5. Nach ca. 1–2 Minuten per SSH verbinden:
   ```
   ssh benutzername@gaushorn-pi.local
   ```

## 3. Projekt auf den Pi übertragen

Vom eigenen Computer aus (Ordner `gaushorn_shot_classifier/` muss lokal vorhanden sein):

```
scp -r gaushorn_shot_classifier benutzername@gaushorn-pi.local:~/
```

Danach auf dem Pi (per SSH) in den Ordner wechseln:

```
cd ~/gaushorn_shot_classifier/deployment
```

## 4. Python-Pakete installieren

Raspberry Pi OS nutzt automatisch **piwheels** (vorkompilierte Pakete für
ARM) — das Kompilieren von scikit-learn von Grund auf dauert sonst sehr
lange, mit piwheels geht es in 1–2 Minuten.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Modell testen (mit einer Beispieldatei)

```
python3 rpi_inference.py "../data/raw/190925 gaushorn/001_p8 10m Holz eigene/oben1.txt"
```

Erwartete Ausgabe:

```
Vorhersage: ECHTER TREFFER (Treffer)  (Entscheidungs-Marge=+0.73, ...)
```

Wenn das funktioniert, ist die Installation korrekt.

## 6. Anschluss an den echten Sensor (Live-Betrieb)

**Wichtig:** Das Skript `rpi_inference.py` erwartet aktuell eine fertige
Textdatei mit 512 Werten (gleiches Format wie die Trainingsdaten). Für den
Live-Betrieb muss man wissen, **wie der Sensor seine Daten überträgt** —
das hängt von der bestehenden Sensor-Hardware der Team ab, zum Beispiel:

- **Seriell / UART** (der Sensor sendet Text- oder Byte-Daten über ein
  USB-Kabel oder die GPIO-Pins): Mit `pyserial` die Daten lesen, dieselben
  512 Werte extrahieren und direkt an `extract_features()` übergeben —
  keine Datei nötig.
- **Eigener Analog-Digital-Wandler (ADC) direkt am Pi** (z. B. MCP3008 über
  SPI, wenn der Pi selbst die Sensor-Spannung misst): Mit `spidev` oder
  `adafruit-mcp3008` die 512 Werte einlesen.
- **Der bestehende Logger schreibt weiterhin .txt-Dateien** (z. B. auf eine
  SD-Karte oder einen Netzwerkordner): Ein einfaches Skript, das den Ordner
  überwacht (`watchdog`-Bibliothek) und bei jeder neuen Datei
  `rpi_inference.py` aufruft.

Da die genaue Schnittstelle der Sensor-Hardware hier nicht bekannt ist,
liefert dieses Projekt nur den Baustein "Rohdaten → Vorhersage"
(`extract_features()` + `predict()` in `rpi_inference.py`). Die Team muss
den Lese-Teil an ihre echte Hardware anpassen — die Modell-Logik selbst
bleibt unverändert.

## 7. Automatischer Start beim Hochfahren (optional, empfohlen)

Damit das Skript nach einem Neustart des Pi automatisch wieder läuft, kann
man einen **systemd-Dienst** einrichten. Beispiel-Datei
`gaushorn-classifier.service` (liegt in diesem Ordner) nach
`/etc/systemd/system/` kopieren und anpassen:

```
sudo cp gaushorn-classifier.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gaushorn-classifier
sudo systemctl start gaushorn-classifier
```

Status und Logs prüfen:

```
sudo systemctl status gaushorn-classifier
journalctl -u gaushorn-classifier -f
```

## 8. Kurz zusammengefasst

| Schritt | Befehl / Aktion |
|---|---|
| OS flashen | Raspberry Pi Imager, SSH + WLAN vorab aktivieren |
| Projekt kopieren | `scp -r gaushorn_shot_classifier ...` |
| Pakete installieren | `pip install -r requirements.txt` (piwheels, schnell) |
| Testen | `python3 rpi_inference.py <beispiel.txt>` |
| Live-Betrieb | Lese-Teil an die echte Sensor-Schnittstelle anpassen |
| Dauerbetrieb | systemd-Dienst einrichten (siehe oben) |
