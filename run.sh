rm -rf lightning_logs
# train vae from ukdale15
python train.py --method vae --houses ukdale15 --apps k
python train.py --method vae --houses ukdale15 --apps m
python train.py --method vae --houses ukdale15 --apps d
python train.py --method vae --houses ukdale15 --apps w
python train.py --method vae --houses ukdale15 --apps f
# train avae of all appliance from ukdale15 
python train.py --method avae --houses ukdale15 --apps kmdwf
# train aada of all appliance from ukdale15
python train.py --method acave --houses ukdale15 --apps kmdwf


# train vae from refit256 
python train.py --method vae --houses refit256 --apps k
python train.py --method vae --houses refit256 --apps m
python train.py --method vae --houses refit256 --apps d
python train.py --method vae --houses refit256 --apps w
python train.py --method vae --houses refit256 --apps f
# train avae of all appliance from refit256 
python train.py --method avae --houses refit256 --apps kmdwf
# train aada of all appliance from refit256 
python train.py --method acave --houses refit256 --apps kmdwf 