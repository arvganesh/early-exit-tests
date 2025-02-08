# early-exit-tests

## Environment Setup
Using Python 3.10:
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
`pip install transformers datasets sentencepiece triton wandb`

Log into wandb and huggingface:
`wandb login`
`huggingface-cli login`

## Training
`python train.py`
