rm -rf lightning_logs
# ./run.sh aada
python train.py --method aada --houses refit256 --apps k
python train.py --method aada --houses refit256 --apps m
python train.py --method aada --houses refit256 --apps d
python train.py --method aada --houses refit256 --apps w
python train.py --method aada --houses refit256 --apps f