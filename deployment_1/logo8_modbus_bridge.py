#!/usr/bin/env python3
"""
Bruecke zwischen Siemens LOGO! 8 (Modbus TCP) und dem Klassifikations-
Modell auf dem Raspberry Pi.

WICHTIG — bitte zuerst lesen (siehe auch EINSATZ_ANLEITUNG.md, Abschnitt 6b):

Die LOGO! 8 ist eine SPS (Kleinsteuerung). Ihr Scan-Zyklus und die Modbus-
Antwortzeit liegen im Bereich von einigen Millisekunden pro Register-
Abfrage. Das reicht NICHT aus, um ein feines 512-Punkte-Signal eines
schnellen Stoss-/Aufprallereignisses (wie in den Trainingsdaten) in
Echtzeit zu uebertragen — das wuerde mehrere Sekunden dauern statt
Millisekunden.

Realistisch sinnvoll ist dieser Weg, wenn die LOGO! 8:
  (a) nur ein Auslöse-Signal ("Ereignis erkannt") per Digitaleingang liefert,
      und/oder
  (b) einen langsameren Analogwert liefert (z. B. Distanzsensor, Spannung).

Wenn die eigentliche 512-Punkte-Signalaufnahme von einem ANDEREN, schnellen
Messgeraet stammt (das vermutlich die urspruenglichen .txt-Dateien erzeugt
hat), dann sollte DIESES Geraet direkt mit dem Raspberry Pi verbunden werden
(seriell/UART, SPI-ADC, oder Datei-Ordner — siehe EINSATZ_ANLEITUNG.md,
Abschnitt 6). Die LOGO! 8 kann in diesem Fall trotzdem parallel als
Ausloese-Quelle dienen (siehe unten).

Voraussetzung auf der LOGO! 8 (in LOGO!Soft Comfort):
  - Netzwerk-Einstellungen: feste IP-Adresse vergeben (gleiches Netz wie der Pi)
  - Im Programm: "Modbus TCP Server" aktivieren (LOGO! 8.2/8.3 FW)
  - Die gewuenschten Werte (z. B. AI1 = analoger Messwert, M1 = Ausloese-Merker)
    auf Modbus-Register legen (Adressen im LOGO!Soft-Comfort-Handbuch nachsehen,
    Kapitel "Modbus TCP")

Installation auf dem Pi:
    pip install pymodbus

Verwendung:
    python3 logo8_modbus_bridge.py --host 192.168.1.50 --trigger-reg 0 --value-reg 1
"""
import argparse
import time
import sys
import os

from pymodbus.client import ModbusTcpClient

# rpi_inference.py liegt im selben Ordner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rpi_inference import extract_features, predict  # noqa: E402


def read_register(client, address, unit=1):
    """Liest ein einzelnes Holding-Register (16 Bit) von der LOGO! 8."""
    result = client.read_holding_registers(address=address, count=1, device_id=unit)
    if result.isError():
        raise IOError(f"Modbus-Lesefehler bei Register {address}: {result}")
    return result.registers[0]


def main():
    ap = argparse.ArgumentParser(description="LOGO! 8 -> Raspberry Pi Modbus-Bruecke")
    ap.add_argument("--host", required=True, help="IP-Adresse der LOGO! 8 (z. B. 192.168.1.50)")
    ap.add_argument("--port", type=int, default=502, help="Modbus-TCP-Port (Standard: 502)")
    ap.add_argument("--trigger-reg", type=int, required=True,
                     help="Register-Adresse des Ausloese-Merkers (0 = kein Ereignis, 1 = Ereignis erkannt)")
    ap.add_argument("--value-reg", type=int, default=None,
                     help="Optional: Register-Adresse eines Analogwerts, der bei jedem Ereignis mitgelesen wird")
    ap.add_argument("--poll-interval", type=float, default=0.2, help="Abfrage-Intervall in Sekunden")
    ap.add_argument("--unit", type=int, default=1, help="Modbus-Geraete-ID (meist 1)")
    args = ap.parse_args()

    client = ModbusTcpClient(args.host, port=args.port)
    if not client.connect():
        print(f"Konnte keine Verbindung zu {args.host}:{args.port} aufbauen.")
        sys.exit(1)
    print(f"Verbunden mit LOGO! 8 auf {args.host}:{args.port}. Beobachte Register {args.trigger_reg} ...")

    was_triggered = False
    try:
        while True:
            trig = read_register(client, args.trigger_reg, unit=args.unit)
            if trig and not was_triggered:
                print("Ereignis erkannt (Ausloese-Merker gesetzt) ...")
                if args.value_reg is not None:
                    val = read_register(client, args.value_reg, unit=args.unit)
                    print(f"  Zusatzwert (Register {args.value_reg}): {val}")
                # Hinweis: Hier liegt noch KEIN vollstaendiges 512-Punkte-Signal
                # vor (siehe Warnhinweis oben). Wenn die LOGO! 8 nur den
                # Trigger liefert, muss das eigentliche Signal von der
                # separaten Erfassungs-Hardware kommen (z. B. per Datei im
                # "eingang"-Ordner, siehe rpi_inference.py --live).
                print("  -> Trigger weitergeleitet. Volles Signal ggf. ueber "
                      "rpi_inference.py --live einlesen.")
            was_triggered = bool(trig)
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nBeendet.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
