python train_pelican_classifier.py \
--datadir=./data/sample_data/run12 \
--target=is_signal \
--nobj=85 \
--nobj-avg=40 \
--num-epoch=64 \
--num-train=60000 \
--num-valid=60000 \
--batch-size=64 \
--prefix=classifier \
--optim=adamw \
--activation=leakyrelu \
--factorize \
--lr-decay-type=warm \
--lr-init=0.0025 \
--lr-final=1e-6 \
--drop-rate=0.05 \
--drop-rate-out=0.05 \
--weight-decay=0.025

# python train_pelican_classifier.py \
# --datadir=/home/meliao/projects/lorentz_group_random_features/data/top_tagging \
# --target=is_signal \
# --nobj=85 \
# --nobj-avg=40 \
# --num-epoch=64 \
# --num-train=1_200_000 \
# --num-valid=60_000 \
# --batch-size=100 \
# --prefix=classifier \
# --optim=adamw \
# --activation=leakyrelu \
# --factorize \
# --lr-decay-type=warm \
# --lr-init=0.0025 \
# --lr-final=1e-6 \
# --drop-rate=0.05 \
# --drop-rate-out=0.05 \
# --weight-decay=0.025


# --datadir=/home/meliao/projects/lorentz_group_random_features/data/top_tagging \
