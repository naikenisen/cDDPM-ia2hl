```bash
module load python
python3 -m venv venv
source venv/bin/activate
pip3 install --prefix=/work/imvia/in156281/cDDPM/venv -r requirements.txt
export PYTHONPATH=/work/imvia/in156281/cDDPM/venv/lib/python3.9/site-packages:$PYTHONPATH
pip3 list

# ajout d'une branche distante 
git pull
git fetch origine
git checkout --track origin/cache
```