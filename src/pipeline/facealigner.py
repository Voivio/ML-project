# https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py

# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import json
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

from data_utils import LFW_merge_file_lists

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def disp_landmarks(img, points):
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    lgs = []
    for k, (s, e) in FACIAL_LANDMARKS_68_IDXS.items():
        plt.scatter(points[s:e, 0], points[s:e, 1], s=4, marker='*')
        lgs.append(k)
    plt.legend(lgs)
    # plt.show()


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def align(image, points, desiredLeftEye=None, desiredFaceWidth=256, desiredFaceHeight=None):
    if desiredLeftEye is None:
        desiredLeftEye = (0.35, 0.35)
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    # simple hack ;)
    assert len(points) == 68
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEyePts = points[lStart:lEnd]
    rightEyePts = points[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    new_points = np.matmul(M, np.concatenate([points.transpose(), np.ones((1, 68), dtype=points.dtype)], axis=0))

    # return the aligned face
    return output, new_points.transpose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--list', nargs='+', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore-error', action='store_true')
    parser.add_argument('--ignore-done', action='store_true')
    parser.add_argument('--desired-left-eye-x', default=0.35, type=float)
    parser.add_argument('--desired-left-eye-y', default=0.35, type=float)
    parser.add_argument('--desired-width', default=256, type=int)
    args = parser.parse_args()

    if args.list is not None:
        jpg_list = LFW_merge_file_lists(args.list, data_root=args.data_root, suffix='.jpg')
    else:
        jpg_list = sorted(list(glob.glob('{}/lfw/*/*_????.jpg'.format(args.data_root))))

    for jpg_fname in tqdm(jpg_list):
        img = cv2.imread(jpg_fname)
        json_fname = jpg_fname.replace('.jpg', '.json')
        # skip invalid data if ignore-error
        if not os.path.exists(json_fname):
            if args.ignore_error:
                print("%s doesn't exist! Skip it." % json_fname)
                continue
            else:
                raise ValueError("%s doesn't exist!" % json_fname)
        # skip done if already transformed
        if os.path.exists(jpg_fname.replace('.jpg', '.transformed.jpg')) and os.path.exists(
                json_fname.replace('.json', '.transformed.json')) and args.ignore_done:
            continue
        # transform
        with open(json_fname) as f:
            data = json.load(f)
        points = np.array(data['landmarks'], dtype=np.int)
        new_img, new_points = align(img, points, desiredLeftEye=(args.desired_left_eye_x, args.desired_left_eye_y),
                                    desiredFaceWidth=args.desired_width)
        if args.debug:
            disp_landmarks(img, points)
            disp_landmarks(new_img, new_points)
            plt.show()
        cv2.imwrite(jpg_fname.replace('.jpg', '.transformed.jpg'), new_img)
        with open(json_fname.replace('.json', '.transformed.json'), 'w') as f:
            json.dump([[float(x), float(y)] for x, y in new_points], f)
