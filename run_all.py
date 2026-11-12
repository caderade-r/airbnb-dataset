"""Run the three sections of the project in order."""
import subprocess
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parent / 'scripts'
for script in ['01_eda.py', '02_train_random_forest.py', '03_evaluate.py']:
    print(f'\nRunning {script}', flush=True)
    subprocess.run([sys.executable, str(scripts / script)], check=True)
print('\nAll results are available in results/.')
