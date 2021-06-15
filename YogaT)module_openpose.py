import cv2
from createDirectory import *

# openpose #1 포인트 검출
def output_keypoints(image_width, image_height, frame, proto_file, weights_file, threshold, model_name, BODY_PARTS):
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의 base : 368
    # 이미지 사이즈를 변경하는 것으로 포인터를 검출하지 못한 이미지 작업을 개선할 수 있다. 보통 255가 잘 나옴.
    #global image_height, image_width
    
    #image_height = 110
    #image_width = 155

    # 네트워크에 넣기 위한 전처리, swapRB를 true로 변경하면 같은 가중치에서도 다른 결과가 검춛된다. (자세한건 추가조사 필요)
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    
    # 포인트 리스트 초기화
    points = []
    
    # 이미지 사이즈 명세용(추후 삭제)
    #print("---원본 사이즈---")
    #print(imag1_size[0], imag1_size[1])
    #print("---out---")
    #print(out.shape)
    #print(out_width, out_height)

    print(f"\n============================== {model_name} Model ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정  // 식 조정으로 좀 더 정확한 포인트를 검출할 수 있는 가능성.
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
            output_filter = True

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

            #output_filter = False
            
    #cv2.imshow("Output_Keypoints", frame)
    cv2.waitKey(1)
    return frame


# In[2]:


# openpose 2 포인트 연결
def output_keypoints_with_lines(pose_name, save_image_count, frame, POSE_PAIRS):

    print("before", save_image_count)
    print()
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            
    #cv2.imshow("output_keypoints_with_lines", frame)
    cv2.waitKey(1)
    save_path = './yoga_images/save_images/'+pose_name+'_result/'
    createDirectory(save_path)
    save_image = save_path + pose_name+'_result (%s).jpg'%save_image_count
    cv2.imwrite(save_image, frame)
    cv2.destroyAllWindows()
    save_image_count += 1
    print("after : ", save_image_count)

    return save_image_count

