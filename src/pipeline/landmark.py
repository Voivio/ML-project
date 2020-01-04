import cv2
import dlib
import os
import json

# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# bring in the input image
data_path = '../../data/lfw/'
people_names = os.listdir(data_path) # because for a image we have
error_name_list = []

for i, people in enumerate(people_names):
    # print('people {} : {} / {}'.format(people, i, len(people_names)))
    img_names = os.listdir(data_path + people)
    for img in img_names:
        if img[-3:] != 'jpg' :
            continue
        json_file = data_path + people + '/' + img[:-3] + 'json'
        # print(pts_file)
        img_gray = cv2.imread(data_path + people + '/' + img, 0)

        # detect faces in the image
        faces_in_image = detector(img_gray, 0)
        try:
            face = faces_in_image[0]
        except IndexError as e:
            error_name_list.append(people + '\t' + str(i) + '\n')
            print('%s'%(img))
            continue

    	# # assign the facial landmarks
        # landmarks = predictor(img_gray, face)
        #
        # # unpack the 68 landmark coordinates from the dlib osbject into a list
        # landmarks_list = []
        # for i in range(0, landmarks.num_parts):
        # 	landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))
        #
        # face = [face.left(), face.top(), face.right(), face.bottom()]
        # dict = {
        #     'face' : face,
        #     'landmarks' : landmarks_list
        # }
        # # dump landmarks
        # with open(json_file, 'w') as f:
        #     json.dump(dict, f)

    	# # for each landmark, plot and write number
    	# for landmark_num, xy in enumerate(landmarks_list, start = 1):
    	# 	cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
    	# 	cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)
