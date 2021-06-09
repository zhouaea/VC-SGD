sed -i 's/^num_training_data:.*/num_training_data: 1000/' config.yml
sed -i 's/^  batch_size:.*/  batch_size: 100/' config.yml
python3.9 main.py

sed -i 's/^  batch_size:.*/  batch_size: 50/' config.yml
python3.9 main.py

sed -i 's/^  batch_size:.*/  batch_size: 25/' config.yml
python3.9 main.py

sed -i 's/^  batch_size:.*/  batch_size: 10/' config.yml
python3.9 main.py