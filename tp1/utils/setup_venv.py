#!/usr/bin/env python3
"""
Créer un environnement virtuel et installer les dépendances depuis requirements.txt
"""

import subprocess
import sys
from pathlib import Path

venv_dir = Path("../venv")

if not venv_dir.exists():
    print("🧱 Création de l'environnement virtuel...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
else:
    print("✅ Environnement virtuel déjà existant.")

if not Path("requirements.txt").exists():
    print("⚠️  Aucun fichier requirements.txt trouvé.")
    sys.exit(1)

pip_path = venv_dir / "bin" / "pip"
print("⬇️ Installation des dépendances depuis requirements.txt...")
subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)

print("\n✅ Environnement prêt !")
print(f"Pour l’activer : source {venv_dir}/bin/activate")
