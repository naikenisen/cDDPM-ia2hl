#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N eval_cDDPM
set -e

cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/cDDPM
source /beegfs/data/work/imvia/in156281/cDDPM/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/cDDPM/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib
export XDG_CACHE_HOME=/work/imvia/in156281/.cache
export TORCH_HOME=/work/imvia/in156281/.cache/torch
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME"
cd
cd /beegfs/data/work/imvia/in156281/cDDPMv2

python eval.py \
	-s /work/imvia/in156281/cDDPMv2/dataset/test/CD30 \
	-d /work/imvia/in156281/cDDPMv2/dataset/test/virtual_CD30_GAN512 \
	--bootstrap-samples 1000 \
	--fid-bootstrap-samples 100 \
	--ci-level 95