# export https_proxy='10.249.36.23:8243'
# conda create -n tair python=3.10 -y
# conda activate tair
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# pip install einops ninja packaging


# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# export https_proxy='10.249.36.23:8243'
# conda create -n tair-cuda12-new python=3.10 -y
# conda activate tair-cuda12-new
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# pip install -r requirements.txt
# apt-get install gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
cd detectron2 
pip install -e .
cd ..
cd testr 
pip install -e .
