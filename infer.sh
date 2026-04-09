#!/bin/ksh
#$ -q gpu
#$ -o infer_result.out
#$ -j y
#$ -N cDDPM_infer

# Paths on cluster
PROJECT_DIR=/beegfs/data/work/imvia/in156281/cDDPMv2
VENV_DIR=/beegfs/data/work/imvia/in156281/cDDPM/venv
TEST_ROOT=/work/imvia/in156281/cDDPMv2/dataset/test
TEST_HES_DIR=$TEST_ROOT/HES
OUT_VIRTUAL_CD30_DIR=$TEST_ROOT/virtual_CD30
RESUME_STATE=/work/imvia/in156281/cDDPMv2/experiments/train_virtual_staining_hes_to_cd30_260326_115613/checkpoint/140
BASE_CONFIG=$PROJECT_DIR/config/config.json
TMP_CONFIG=$PROJECT_DIR/config/config.infer_140.json

cd $WORKDIR
module load python
source $VENV_DIR/bin/activate

export PYTHONPATH=/work/imvia/in156281/cDDPM/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

cd $PROJECT_DIR

# Build a dedicated inference config with test data root + checkpoint 140
python - <<'PY'
import json
from pathlib import Path

base_config = Path('/beegfs/data/work/imvia/in156281/cDDPMv2/config/config.json')
out_config = Path('/beegfs/data/work/imvia/in156281/cDDPMv2/config/config.infer_140.json')

test_root = '/work/imvia/in156281/cDDPMv2/dataset/test'
resume_state = '/work/imvia/in156281/cDDPMv2/experiments/train_virtual_staining_hes_to_cd30_260326_115613/checkpoint/140'

cfg = json.loads(base_config.read_text())
cfg['datasets']['test']['which_dataset']['args']['data_root'] = test_root
cfg['datasets']['test']['which_dataset']['args'].pop('allowed_patches', None)
cfg['path']['resume_state'] = resume_state

out_config.write_text(json.dumps(cfg, indent=4))
print(f'Wrote {out_config}')
print(f"test data_root: {cfg['datasets']['test']['which_dataset']['args']['data_root']}")
print(f"resume_state: {cfg['path']['resume_state']}")
PY

python run.py -c $TMP_CONFIG -p test

# Collect only generated virtual stains into test/virtual_CD30
python - <<'PY'
from pathlib import Path
import shutil

test_hes_dir = Path('/work/imvia/in156281/cDDPMv2/dataset/test/HES')
out_virtual_dir = Path('/work/imvia/in156281/cDDPMv2/dataset/test/virtual_CD30')
experiments_root = Path('/beegfs/data/work/imvia/in156281/cDDPMv2/experiments')

if not test_hes_dir.exists():
    raise FileNotFoundError(f'Missing HES directory: {test_hes_dir}')

test_runs = sorted(experiments_root.glob('test_virtual_staining_hes_to_cd30_*'))
if not test_runs:
    raise FileNotFoundError('No test run directory found under experiments/')

latest_test_run = test_runs[-1]
test_result_root = latest_test_run / 'results' / 'test'
if not test_result_root.exists():
    raise FileNotFoundError(f'No test results found in {test_result_root}')

epoch_dirs = sorted([p for p in test_result_root.iterdir() if p.is_dir()])
if not epoch_dirs:
    raise FileNotFoundError(f'No epoch result directory found in {test_result_root}')

latest_epoch_dir = epoch_dirs[-1]
out_virtual_dir.mkdir(parents=True, exist_ok=True)

generated = list(latest_epoch_dir.glob('Out_*'))
if not generated:
    raise FileNotFoundError(f'No generated Out_* files found in {latest_epoch_dir}')

copied = 0
for f in generated:
    target_name = f.name[len('Out_'):]
    shutil.copy2(f, out_virtual_dir / target_name)
    copied += 1

print(f'Latest test run: {latest_test_run}')
print(f'Source generated dir: {latest_epoch_dir}')
print(f'Copied {copied} files to {out_virtual_dir}')
PY
