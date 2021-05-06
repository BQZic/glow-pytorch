HPARAMS="hparams"
DATASET_ROOT="/content/drive/MyDrive/CelebA/img_align_celeba"
Z_DIR="celeba_z"

python3 infer_celeba.py ${HPARAMS} ${DATASET_ROOT} ${Z_DIR}