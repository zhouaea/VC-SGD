gsed -i 's/^num_training_data:.*/num_training_data: 1000/' config.yml
gsed -i 's/^  batch_size:.*/  batch_size: 100/' config.yml
python3.9 main.py

gsed -i 's/^  batch_size:.*/  batch_size: 50/' config.yml
python3.9 main.py

gsed -i 's/^  batch_size:.*/  batch_size: 25/' config.yml
python3.9 main.py

gsed -i 's/^  batch_size:.*/  batch_size: 10/' config.yml
python3.9 main.py