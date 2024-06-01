rm -rf lightning_logs
# run for all appliances in ukdale
# ./run.sh aada
nohup python train.py --method $1 --houses ukdale15 --apps k &
nohup python train.py --method $1 --houses ukdale15 --apps m &
nohup python train.py --method $1 --houses ukdale15 --apps d &
nohup python train.py --method $1 --houses ukdale15 --apps w &
nohup python train.py --method $1 --houses ukdale15 --apps f &

# # run for all appliances in refit
# nohup python train.py --method $1 --houses refit256 --apps k &
# nohup python train.py --method $1 --houses refit256 --apps m &
# nohup python train.py --method $1 --houses refit256 --apps d &
# nohup python train.py --method $1 --houses refit256 --apps w &
# nohup python train.py --method $1 --houses refit256 --apps f &