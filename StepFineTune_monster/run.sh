srun --partition amd-longq --nodes 1 --gres=gpu python gan.py
python discriminator_final.py data
