conda create -n paddle python=3.10
conda activate paddle
python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr
python paddle_ocr.py
