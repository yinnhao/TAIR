import argparse
import os
from glob import glob
from paddleocr import PaddleOCR

def main():
    parser = argparse.ArgumentParser(description='PaddleOCR script')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # Create output directory if not exists
    os.makedirs(args.output, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Handle single file or directory
    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = glob(os.path.join(args.input, '*'))

    for file_path in files:
        result = ocr.predict(input=file_path)
        for res in result:
            res.print()
            res.save_to_img(args.output)
            res.save_to_json(args.output)

if __name__ == '__main__':
    main()