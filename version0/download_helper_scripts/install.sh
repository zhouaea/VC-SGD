sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install python3.8-venv
python3.8 -m venv env
pip install -r requirements.txt # Install dependencies