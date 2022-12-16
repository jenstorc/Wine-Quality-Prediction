Start virtual environnement :
python3.8 -m venv .venv
source .venv/bin/activate
conda deactivate
pip install --upgrade pip
pip install -r requirements.txt

Start server : 
uvicorn __init__:app --reload