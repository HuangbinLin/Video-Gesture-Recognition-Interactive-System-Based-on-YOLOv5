import cv2
import numpy as np
import argparse
import onnxruntime as ort
from  matplotlib import pyplot as plt


class yolov5():
    def __init__(self, model_pb_path, confThreshold=0.9, nmsThreshold=0.5, objThreshold=0.5):
        self.net = ort.InferenceSession(model_pb_path)
        self.classes = ['invalid', 'up', 'down', 'left', 'right', 'close']
        self.num_classes = len(self.classes)
        anchors = [[4,4, 8,8, 11,11], [15,15, 20,22, 26,27], [34,36, 48,49, 84,86]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = self.num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        height, width = srcimg.shape[:2]

        # 计算放缩比例
        ratio = min(neww / width, newh / height)
        resized_width = int(width * ratio)
        resized_height = int(height * ratio)

        # resize
        resized_image = cv2.resize(srcimg, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        # 对其余地方进行黑色填充
        pad_width = neww - resized_width
        pad_height = newh - resized_height
        padh = pad_height // 2
        padw = pad_width // 2

        img = np.zeros((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8)
        img[padh:padh+resized_height, padw:padw+resized_width] = resized_image
        return img, resized_height, resized_width, padh, padw

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                classIds.append(classId)
                confidences.append(float(confidence))

        box_index = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        #frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        # filtered_boxes = [boxes[i[0]] for i in box_index]
        # filtered_classIds = [classIds[i[0]] for i in box_index]
        # filtered_confidences = [confidences[i[0]] for i in box_index]

        filtered_boxes = [boxes[i] for i in box_index]
        filtered_classIds = [classIds[i] for i in box_index]
        filtered_confidences = [confidences[i] for i in box_index]
        
        return filtered_boxes,filtered_classIds,filtered_confidences

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), thickness=1)
        return frame

    def detect(self, srcimg):
        img, resized_height, resized_width, padh, padw = self.resize_image(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        boxes,classIds,confidences = self.postprocess(srcimg, outs, (resized_height, resized_width, padh, padw))
        return boxes,classIds,confidences