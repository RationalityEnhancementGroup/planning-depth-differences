module load anaconda/3/2021.11

conda create --name irl-project python=3.8
conda activate irl-project

conda install -c r r
python -m pip install -r requirements-cluster.txt
