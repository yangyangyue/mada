rm -rf lightning_logs
# ./run.sh aada
nohup python train.py --method aada --houses ukdale15 --apps k > log/uk  2>&1 &
nohup python train.py --method aada --houses ukdale15 --apps m > log/um  2>&1 &
nohup python train.py --method aada --houses ukdale15 --apps d > log/ud  2>&1 &
nohup python train.py --method aada --houses ukdale15 --apps w > log/uw  2>&1 &
nohup python train.py --method aada --houses ukdale15 --apps f > log/uf  2>&1 &

nohup python train.py --method aada --houses refit256 --apps k > log/rk  2>&1 &
nohup python train.py --method aada --houses refit256 --apps m > log/rm  2>&1 &
nohup python train.py --method aada --houses refit256 --apps d > log/rd  2>&1 &
nohup python train.py --method aada --houses refit256 --apps w > log/rw  2>&1 &
nohup python train.py --method aada --houses refit256 --apps f > log/rf  2>&1 &
