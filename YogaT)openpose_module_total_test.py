
import cv2
import os
from PIL import Image
import numpy as np

from module_yolo import *
from module_resize import *
from module_openpose import *

# 입력 이미지의 사이즈 정의 base : 368
# 이미지 사이즈를 변경하는 것으로 포인터를 검출하지 못한 이미지 작업을 개선할 수 있다. 보통 255가 잘 나옴.
image_height = 368
image_width = 368

# 특정 조건이 성립할 때 결과출력을 중단하도록 지정.
output_filter = True

#이미지 저장 넘버링 용 카운터 (좌우반전시 카운터중복 방지용 별도 카운터)
save_image_count = 1


# YOLO crop -> 전처리 -> openpose 검출 후 결과를 이미지 파일로 저장 순으로 모듈 통합 및 실행 테스트

# 통합 모듈 반복 테스트
# 일괄처리 하고 싶은 경우 활성화.
pose_name = input("input name of pose : ")
pose_name = pose_name.capitalize()
#image_width = int(input("input width of image : "))
#image_height = int(input("input height of image : "))

# 전처리용 사이즈 정의
sizX = input("input size x : ")
sizY = input("input size y : ")
size = (int(sizX), int(sizY))

size = input("input a size of border : ")

#모듈별 디렉토리 기준, 명세 등
# 자세선택
yoga_path = r"./yoga_images/"+pose_name+"_pose/"

# YOLO 모듈, 전처리 모듈 실행 후 저장 경로
modified_path = r"./yoga_images/resize_image/"+pose_name+"_resize/"

# resize 모듈 대상 경로
resize_path = r"./yoga_images/resize_image/"+pose_name+"_resize/"

#경로내 파일의 갯수
files = os.listdir(yoga_path)
dir_len = len(files)
print(dir_len)


# 특정 이미지만 사용하고 싶은 경우 반복문 파라미터 조정, 기본치는 1에서 경로내 이미지 전체 대상
for x in range(1, dir_len + 1):
    count = x
    #1 YOLO
    #YOLO_image(pose_name, yoga_path, modified_path, count)

    #2 전처리
    # 전처리할 이미지 경로


    resize_image = resize_path + pose_name + " (%s).jpg" % count

    #이미지 원본 기준으로 전처리, 원하는 규격을 줄때는 resize와 border의 순서를 반대로
    image1 = Image.open(resize_image)
    imag1_size = image1.size
    size = (int(imag1_size[0]), int(imag1_size[1]))

    resize_and_crop(resize_image, pose_name, modified_path, size, count, crop_type='middle')
    set_border(resize_image, pose_name, modified_path, count)


    #3 openpose
    BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                      10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                      15: "Background"}

    POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                      [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

    BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                       5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                       10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                       15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

    POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                       [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

    BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                          5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                          10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                          15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                          20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

    POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                          [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                          [11, 24], [22, 24], [23, 24]]

    # 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_mpi = "./openpose-master/models/pose/mpi/pose_deploy_linevec.prototxt"
    protoFile_mpi_faster = "./openpose-master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    protoFile_coco = "./openpose-master/models/pose/coco/pose_deploy_linevec.prototxt"
    protoFile_body_25 = "./openpose-master/models/pose/body_25/pose_deploy.prototxt"

    # 훈련된 모델의 weight 를 저장하는 caffemodel 파일
    weightsFile_mpi = "./openpose-master/models/pose/mpi/pose_iter_160000.caffemodel"
    weightsFile_coco = "./openpose-master/models/pose/coco/pose_iter_440000.caffemodel"
    weightsFile_body_25 = "./openpose-master/models/pose/body_25/pose_iter_584000.caffemodel"

    # 키포인트를 저장할 빈 리스트
    points = []

    man = resize_path + pose_name + " (%s).jpg" %count

    frame_mpii = cv2.imread(man)
    frame_mpii2 = cv2.flip(frame_mpii, 1)


    # image_height = int(imag1_size[1] /2)
    # image_width = int(imag1_size[0] /2)


    # 이미지 분석
    frame_MPII = output_keypoints(image_width, image_height, frame=frame_mpii, proto_file=protoFile_mpi, weights_file=weightsFile_mpi,
                                  threshold=0.01, model_name="MPII", BODY_PARTS=BODY_PARTS_MPI)
    save_image_count = output_keypoints_with_lines(pose_name, save_image_count, frame=frame_MPII, POSE_PAIRS=POSE_PAIRS_MPI)

    # 좌우반전된 이미지 분석
    frame_MPII = output_keypoints(image_width, image_height, frame=frame_mpii2, proto_file=protoFile_mpi, weights_file=weightsFile_mpi,
                                  threshold=0.01, model_name="MPII", BODY_PARTS=BODY_PARTS_MPI)
    save_image_count = output_keypoints_with_lines(pose_name, save_image_count, frame=frame_MPII, POSE_PAIRS=POSE_PAIRS_MPI)


print("finish")
save_image_count = 1
