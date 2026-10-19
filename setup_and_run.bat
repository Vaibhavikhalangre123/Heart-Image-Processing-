@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_arcade.py --subset stenosis --split train --limit 10
python process_arcade.py --input-dir data\ARCADE_processed\stenosis\train\images --output-dir outputs\arcade\stenosis_train
python benchmark.py --input-dir data\ARCADE_processed\stenosis\train\images --warmup 20 --iterations 100
pause
