import cv2
import numpy as np

test_text_instance = {    # 每个text_instance是一个字典，有三个key: "bbox"、"text"、"polygon"
          "bbox": [  # 矩形框 ：左上和右下坐标 x1,y1,x2,y2
            255,
            450,
            279,
            460
          ],
          "text": "CHO",  # 文本
          "polygon": [  # 多边形
            [
              255.63059997558594,
              450.606201171875
            ],
            [
              258.453369140625,
              450.83929443359375
            ],
            [
              261.9189453125,
              450.8673400878906
            ],
            [
              265.2309875488281,
              450.9798278808594
            ],
            [
              268.7821044921875,
              450.95880126953125
            ],
            [
              272.4165954589844,
              450.85614013671875
            ],
            [
              275.567138671875,
              450.8016052246094
            ],
            [
              278.77020263671875,
              450.8444519042969
            ],
            [
              278.5768127441406,
              459.4132995605469
            ],
            [
              275.5859069824219,
              459.6985778808594
            ],
            [
              272.1300354003906,
              459.77850341796875
            ],
            [
              268.869873046875,
              459.62982177734375
            ],
            [
              265.3174133300781,
              459.3565368652344
            ],
            [
              261.8007507324219,
              459.2027282714844
            ],
            [
              258.17578125,
              458.9566650390625
            ],
            [
              255.29116821289062,
              458.7799377441406
            ]
          ]
        }
text = "abc"
# 处理边界框
box_xyxy = test_text_instance['bbox']
x1,y1,x2,y2 = box_xyxy
# box_xyxy_scaled = list(map(lambda x: x/model_H, box_xyxy))  # scale box coord to [0,1]
# x1,y1,x2,y2 = box_xyxy_scaled 
box_xywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]   # xyxy -> cxcywh
# box format
# processed_box = box_cxcywh
# processed_box = list(map(lambda x: round(x,4), processed_box))
# boxes.append(processed_box)


# 处理多边形
poly = np.array(test_text_instance['polygon']).astype(np.int32)    # 16 2
# scale poly
# poly_scaled = poly / np.array([model_W, model_H])
# polys.append(poly_scaled)

# path = ""
gt_path = "/root/paddlejob/workspace/env_run/zhuyinghao/datasets/text_v1/images/ip11pro_output_testSR_2101225_xwm_renew_0.JPG"
# VISUALIZE FOR DEBUGGING
img0 = cv2.imread(gt_path)  # 512 512 3
x, y, w, h = box_xywh
# img0_box = cv2.rectangle(img0, (x,y), (x+w, y+h), (0,255,0), 2)
img0_poly = cv2.polylines(img0, [poly], True, (0,255,0), 2)
# cv2.putText(img0_box, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
cv2.putText(img0_poly, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
# cv2.imwrite('./img0_box.jpg', img0_box)
cv2.imwrite('./img0_poly.jpg', img0_poly)
