import cv2
import dlib
import os
import pickle
import json

# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# bring in the input image
data_path = '../../data/lfw/'
# people = 'Giulietta_Masina'
# img_names = os.listdir(data_path + people)
# img = img_names[0]
# img_color = cv2.imread(data_path + people + '/' + img, 1)
# img_gray = cv2.imread(data_path + people + '/' + img, 0)
#
# with open(data_path + people + '/' + img[:-3] + 'pkl', 'rb') as f:
#     info = pickle.load(f)
#
# print(info['face'].left)
# print(info['landmarks'])

# # detect faces in the image
# faces_in_image = detector(img_gray, 0)
# print(len(faces_in_image))
# face = faces_in_image[0]
# # for face in faces_in_image:
#     # assign the facial landmarks
#     landmarks = predictor(img_gray, face)

    # # unpack the 68 landmark coordinates from the dlib object into a list
    # landmarks_list = []
    # for i in range(0, landmarks.num_parts):
    #     landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

# for each landmark, plot and write number
# for landmark_num, xy in enumerate(info['landmarks'], start = 1):
#     cv2.circle(img_color, (xy[0], xy[1]), 12, (168, 0, 20), -1)
#
# cv2.rectangle(img_color, (info['face'].left(), info['face'].top()), (info['face'].right(), info['face'].bottom()), (0,0,255), 2)
#
# cv2.imwrite('img.png', img_color)

# dict = {
#     'face' : [1,2,3,4],
#     'landmarks' : [(1,2), (3,4)]
# }
#

with open('nofacelist.txt', 'r') as f:
    for line in f:
        # print(line[:-1].split('_'))
        pieces = line[:-1].split('_')
        people = '_'.join(pieces[:-1])
        json_file = data_path + people + '/' + line[:-4] + 'json'
        # print(data_path + people + '/' + line[:-1])
        img = cv2.imread(data_path + people + '/' + line[:-1])
        faces_in_image = detector(img, 0)

        try:
            face = faces_in_image[0]
        except IndexError as e:
            # error_name_list.append(people + '\t' + str(i) + '\n')
            print('%s'%(line[:-1]))
            continue

        landmarks = predictor(img, face)

        # unpack the 68 landmark coordinates from the dlib osbject into a list
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
        	landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        face = [face.left(), face.top(), face.right(), face.bottom()]
        dict = {
            'face' : face,
            'landmarks' : landmarks_list
        }
        # dump landmarks
        with open(json_file, 'w') as f:
            json.dump(dict, f)
