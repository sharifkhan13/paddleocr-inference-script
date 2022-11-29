from turtle import width
from paddleocr import PaddleOCR, draw_ocr
import cv2
import time
import os
import pathlib
from PIL import Image, ImageDraw, ImageFont
import shutil

#for test purpose only
# import copy
# bounding_box={
#     "X1": 0.0, #ocr_position_x
#     "X2": 0.0+1.0, # ocr_position_x + ocr_width
#     "Y1": 0.6, #ocr_position_y
#     "Y2": 0.6+1.0 # ocr_position_y + ocr_height
# }


# def get_image_height(img):
#     return img.shape[0]

# def get_image_width(img):
#     return img.shape[1]


# def get_cropped_img(img_path):
#     img=cv2.imread(img_path)
#     y1=round(get_image_height(img)*bounding_box["Y1"])
#     y2=round(get_image_height(img)*bounding_box["Y2"])
#     x1=round(get_image_width(img)*bounding_box["X1"])
#     x2=round(get_image_width(img)*bounding_box["X2"])
#     img=img[y1:y2, x1:x2]
#     return img

class Paddle:
    def __init__(self, det_model, rec_model=None, cls_model=None) -> None:
        self.det_path=os.path.join(pathlib.Path(__file__).parent.absolute(), f"models/det/{det_model}/")
        self.rec_path=os.path.join(pathlib.Path(__file__).parent.absolute(), f"models/rec/{rec_model}/")
        self.cls_path=os.path.join(pathlib.Path(__file__).parent.absolute(), f"models/cls/{cls_model}/")
        # self.image_path=os.path.join(pathlib.Path(__file__).parent.absolute(), f"img/{image_name}")
        self.image_path=os.path.join(pathlib.Path(__file__).parent.absolute(), "img/")
        self.font_path=os.path.join(pathlib.Path(__file__).parent.absolute(), "font/Oswald-Bold.ttf")
        # self.output=os.path.join(pathlib.Path(__file__).parent.absolute(),f"output/image_{image_name.split('.')[0]}")
        self.output=os.path.join(pathlib.Path(__file__).parent.absolute(),"output")
        # self.image_name=image_name
        self.det_model_name=det_model

    def get_result(self):
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=self.det_path,use_gpu=False,
         use_dilation=True, det_db_score_mode='slow', show_log=True, rec_model_dir=self.rec_path, cls_model_dir=self.cls_path)
        start = time.time()
        # result = ocr.ocr(self.image_path, cls=True)
        

        for image in os.listdir(self.image_path):
            output_dir=os.path.join(self.output, f"image_{image.split('.')[0]}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path=os.path.join(output_dir, f"{self.det_model_name}_{image}")
            input_path=os.path.join(self.image_path, image)
            result = ocr.ocr(input_path, cls=True)
            # # for testing purpose only
            # img=get_cropped_img(input_path)
            # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # #width=int(img.shape[1] * 1.75) #it could be works
            # # height=int(img.shape[0] * 1.75)
            # # dim=(width, height)
            # # img=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            # #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img=cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_AREA)
            # copy_img=copy.copy(img)
            # gray=cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY)
            # blur=cv2.GaussianBlur(gray, (5,5), 0)
            # #thresh_otsu=cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]
            # thresh=cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # #thresh_bin=cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # #cv2.imshow("gray", gray)
            # #cv2.imshow("blur", blur)
            # # cv2.imshow("thresh", thresh)
            # # cv2.imshow("thresh_otsu", thresh_otsu)
            # # cv2.imshow("THRESH_BINARY", thresh_bin)
            # # cv2.waitKey(0)
            # result = ocr.ocr(thresh, cls=False)

            
            # will be uncommand
            img = Image.open(input_path).convert('RGB')
            boxes = [line[0] for line in result]
            texts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(img, boxes, texts, scores, font_path=self.font_path)
            im_show = Image.fromarray(im_show)
            im_show.save(output_path)
            original_image=os.path.join(output_dir, image)
            if not os.path.exists(original_image):
                shutil.copyfile(input_path, original_image)
        
        end = time.time()
        print("total time: ", end - start)
        
# # ocr = PaddleOCR(use_angle_cls=True, lang='en')

# font_path=os.path.join(pathlib.Path(__file__).parent.absolute(),
#                             "font/Oswald-Bold.ttf")
# output=os.path.join(pathlib.Path(__file__).parent.absolute(),
#                             "output/paddle.png")

# # BASE_DIR_DET = os.path.join(pathlib.Path(__file__).parent.absolute(),
# #                             "paddle_ocr_own_model/det/own_v3")

# BASE_DIR_DET = os.path.join(pathlib.Path(__file__).parent.absolute(),
#                             "paddle_ocr_pre_train_models2/det/en_PP-OCRv3_det_infer")
# BASE_DIR_REC = os.path.join(pathlib.Path(__file__).parent.absolute(),
#                             "paddle_ocr_pre_train_models2/rec/en_PP-OCRv3_rec_infer")
# BASE_DIR_CLS = os.path.join(pathlib.Path(__file__).parent.absolute(),
#                             "paddle_ocr_pre_train_models2/cls/ch_ppocr_mobile_v2.0_cls_infer")

# ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=BASE_DIR_DET, rec_model_dir=BASE_DIR_REC,
#                 cls_model_dir=BASE_DIR_CLS)

# img_path = "img/ski_2.png"
# img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# start = time.time()
# result = ocr.ocr(img_path, cls=True)
# end = time.time()
# print("total time: ", end - start)
# for res in result:
#     print(res)
# # print(result)


# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# texts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]

# font = ImageFont.load_default()

# im_show = draw_ocr(image, boxes, texts, scores, font_path=font_path)
# im_show = Image.fromarray(im_show)
# im_show.save(output)