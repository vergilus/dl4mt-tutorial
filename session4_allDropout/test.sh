THEANO_FLAGS=device=gpu0 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT02.cn MT02.trans >MT02.log &
THEANO_FLAGS=device=gpu1 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT03.cn MT03.trans >MT03.log &
THEANO_FLAGS=device=gpu2 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT06.cn MT06.trans >MT06.log &
THEANO_FLAGS=device=gpu3 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT08.cn MT08.trans >MT08.log 
THEANO_FLAGS=device=gpu0 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT08NW.cn MT08NW.trans >MT08NW.log &
THEANO_FLAGS=device=gpu1 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT08WB.cn MT08WB.trans >MT08WB.log &
THEANO_FLAGS=device=gpu2 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT04.cn MT04.trans >MT04.log &
THEANO_FLAGS=device=gpu3 python -u  translate_gpu.py -n model_memDec.npz ../data/cn.txt.pkl ../data/en.txt.pkl ../data/MT05.cn MT05.trans >MT05.log 