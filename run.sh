rm -rf lightning_logs

python train.py --method acvae --houses ukdale15 --apps kmdwf
python train.py --method acvae --houses ukdale15 --apps mdwf
python train.py --method acvae --houses ukdale15 --apps kdwf
python train.py --method acvae --houses ukdale15 --apps kmwf
python train.py --method acvae --houses ukdale15 --apps kmdf
python train.py --method acvae --houses ukdale15 --apps kmdw
python train.py --method acvae --houses refit256 --apps kmdwf
python train.py --method acvae --houses refit256 --apps mdwf
python train.py --method acvae --houses refit256 --apps kdwf
python train.py --method acvae --houses refit256 --apps kmwf
python train.py --method acvae --houses refit256 --apps kmdf
python train.py --method acvae --houses refit256 --apps kmdw


