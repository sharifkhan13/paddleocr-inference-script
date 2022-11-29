import argparse
from ocr import Paddle



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-fp', "--font_path", type=str, help="output font path")
    # parser.add_argument('-op', "--output_path", type=str, help="result save path")
    parser.add_argument('-dm', "--det_model", type=str,help="text detection model name")
    parser.add_argument('-rm', "--rec_model", type=str,help="text recognization model name")
    parser.add_argument('-cm', "--cls_model", type=str,help="angle classifier model name")
    # parser.add_argument('-i', "--image_name", type=str,help="input image name")

    args = vars(parser.parse_args())

    if args.get("det_model") is None or args.get("rec_model") is None or args.get("cls_model") is None:
        exit("detector/recognizer/clssifer path is required")
    

    det_model = args.get("det_model")
    rec_model= args.get("rec_model")
    cls_model = args.get("cls_model")
    # image_name= args.get("image_name")

    ocr = Paddle(det_model=det_model, rec_model=rec_model, cls_model=cls_model)
    ocr.get_result()