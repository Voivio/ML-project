import cv2

from ESRTesting import applyModel

for img_name in img_names:
    img = cv2.imread(img_name)

    box, points, succeeded = applyModel(img, model)

    if succeeded:
        dict = {
            'face': box,
            'landmarks': points
        }
        with open(json_file, 'w') as f:
            json.dump(dict, f)
    else:
        continue
