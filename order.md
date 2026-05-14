python main.py --models pet --mode train --output_dir ../tests --epochs 1

python main.py --models iqformer --mode train --output_dir ../tests --epochs 1

--dataset 2016b

source activate radioml && python script/finetune_experiments.py --task 2 --sigma_err 3.0 --finetune_epochs 50 --finetune_lr 1e-4 --batch_size 128



