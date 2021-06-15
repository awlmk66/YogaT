
import cv2
from createDirectory import *
import numpy as np


def YOLO_image(pose_name, yoga_path, modified_path, count):

    # Yolo 로드 -> 테스트 후 def 함수로 뺄 예정
    net = cv2.dnn.readNet("./YOLO/yolov3.weights", "./YOLO/yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    img = cv2.imread(yoga_path + pose_name + " (%s).jpg"%count)
    img = cv2.resize(img, None, fx=1, fy=1)     # 원본에서의 비율 조정 가능
    height, width, channels = img.shape

    # Detecting objects
    # 그 외 사이즈
    # 320 × 320 : 적은 정확도, 빠른 속도
    # 416 × 416 : 중간
    # 609 × 609 : 높은 정확도, 느린 속도 (blob 내 파라미터)

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.95:  # confidence : 0~1의 값. 1에 가까울 수록 정확도가 상승, 객체인지 숫자 하강. 반대의 경우 반비례
                # Object detected
                center_x = int((detection[0] * width)*1.1)
                center_y = int((detection[1] * height)*1.1)
                # w, h *1.1 등을 통해 box의 너비를 넓힐 수 있으나 object box가 이미지를 벗어나는 에러 유의. 
                w = int((detection[2] * width))
                h = int((detection[3] * height))
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 이미지에서 노이즈 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    #cv2.imshow("Image", img)


    # 이미지 크롭 : 좌표기준으로 자르기(x, y, w, h = YOLO에서 받아온 좌표값)
    crop_img = img[y:y+h, x:x+w] # Crop from x, y, w, h -> 100, 200, 300, 400
    #cv2.imshow("Image", crop_img)

    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    createDirectory(modified_path)
    cv2.imwrite(modified_path + pose_name +" ("+str(count)+").jpg", crop_img)
    print(count)




