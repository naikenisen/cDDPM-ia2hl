#!/bin/ksh
#$ -q gpu
#$ -o infer_result.out
#$ -j y
#$ -N cDDPM_infer

# Paths on cluster
PROJECT_DIR=/beegfs/data/work/imvia/in156281/cDDPMv2
VENV_DIR=/beegfs/data/work/imvia/in156281/cDDPM/venv
TEST_ROOT=/work/imvia/in156281/cDDPMv2/dataset/test
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
cfg['path']['resume_state'] = resume_state

out_config.write_text(json.dumps(cfg, indent=4))
print(f'Wrote {out_config}')
print(f"test data_root: {cfg['datasets']['test']['which_dataset']['args']['data_root']}")
print(f"resume_state: {cfg['path']['resume_state']}")
PY

python run.py -c $TMP_CONFIG -p test
