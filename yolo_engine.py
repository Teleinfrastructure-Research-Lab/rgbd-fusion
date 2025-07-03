import sys
import os
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml('coco128.yaml'))['names']


# model = YOLO('yolov8n-seg.engine')
model = YOLO('yolov8n-seg.engine')


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Holotwin.Stream.FileReader import FileReader

height = 424
width = 512
channels = 3

delay = 0
count = 0

def process(input_image):

    input_image = cv2.resize(input_image, (640, 640))
    results = model(input_image, device = 0)
    result = results[0]
    actors = []
    bb = np.array(result.boxes.xyxy.cpu(), dtype = "int")
    cls = np.array(result.boxes.cls.cpu(), dtype = "int")
    scores = np.array(result.boxes.conf.cpu(), dtype = "float")

    if result.masks != None:
        for i, mask in enumerate(result.masks.data):
            if cls[i] == 0 and scores[i] > 0.6:
                actors.append(mask.cpu())

    global count
    global delay
    if count > 0:
        delay += (result.speed["preprocess"] + result.speed["inference"] + result.speed["postprocess"])
        count += 1
    else:
        count += 1

    return input_image, actors



with FileReader("data/trainCap2person.txt") as fr:
    for frame in fr:
        print(frame.header)
        xyz_image = frame.vertexAttributes[0].data.reshape(height, width, channels)
        xyz_image = np.nan_to_num(xyz_image,   nan=0.0, posinf=1, neginf=0.0)
        min_value = xyz_image.min()
        max_value = xyz_image.max() 
        xyz_image_norm = (xyz_image - min_value) / (max_value - min_value)
        rgb_image = frame.vertexAttributes[1].data.reshape(height, width, channels).copy()

        output_image, actors = process(rgb_image)

        for actor in actors:
            output_image[actor == 1, 0] = 255
        cv2.imshow('Image Sequence', output_image)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

print(delay/count)
cv2.destroyAllWindows()