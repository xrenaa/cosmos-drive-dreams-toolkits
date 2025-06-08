# Cosmos Drive Dreams


## Installation
```bash
# Create the cosmos-transfer1 conda environment.
conda env create --file environment.yaml
# Activate the cosmos-transfer1 conda environment.
conda activate cosmos-drive-dreams
# Install the dependencies.
pip install -r requirements.txt
# Install vllm
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
export VLLM_ATTENTION_BACKEND=FLASHINFER
pip install vllm==0.9.0

# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.12
# Install Transformer engine.
pip install transformer-engine[pytorch]==2.4.0
```
