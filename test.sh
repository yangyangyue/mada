# train vae from ukdale15

python test.py --method $1 --houses ukdale1 --apps k --ckpt ${$1}-refit256-k
python test.py --method $1 --houses ukdale1 --apps m --ckpt ${$1}-refit256-m
python test.py --method $1 --houses ukdale1 --apps d --ckpt ${$1}-refit256-d
python test.py --method $1 --houses ukdale1 --apps w --ckpt ${$1}-refit256-w
python test.py --method $1 --houses ukdale1 --apps f --ckpt ${$1}-refit256-f

python test.py --method $1 --houses ukdale2 --apps k --ckpt ${$1}-refit256-k
python test.py --method $1 --houses ukdale2 --apps m --ckpt ${$1}-refit256-m
python test.py --method $1 --houses ukdale2 --apps d --ckpt ${$1}-refit256-d
python test.py --method $1 --houses ukdale2 --apps w --ckpt ${$1}-refit256-w
python test.py --method $1 --houses ukdale2 --apps f --ckpt ${$1}-refit256-f

python test.py --method $1 --houses ukdale5 --apps k --ckpt ${$1}-refit256-k
python test.py --method $1 --houses ukdale5 --apps m --ckpt ${$1}-refit256-m
python test.py --method $1 --houses ukdale5 --apps d --ckpt ${$1}-refit256-d
python test.py --method $1 --houses ukdale5 --apps w --ckpt ${$1}-refit256-w
python test.py --method $1 --houses ukdale5 --apps f --ckpt ${$1}-refit256-f
