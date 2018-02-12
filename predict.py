from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import os
import time

import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--weights', default='/Users/abhishek/Downloads/ssd300_0712_10000.pth',
#                     type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    # print(net)
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    
    videoFilePath = "/home/abhishek/data/dataset/video_test.avi"
    print('home/abhishek/data/dataset/video_test.avi')
    videoFile = cv2.VideoCapture(videoFilePath)
    array = []
    i = 0
    while True:
        ret, image = videoFile.read()
        # print(ret, image)
        print("printing")
        i = str(i)
        output = predict(image)
        print("---------------x")
        save = os.path.join('/home/abhishek/data/dataset/images', i + '.png')
        cv2.imwrite(save,output)

        # img = cv2.imread('/mnt/sdc1/abhishek/1.png')
        # height , width , layers =  img.shape
        # video = cv2.VideoWriter('/mnt/sdc1/abhishek/video.mp4',-1,1,(width,height))
        # print("cool")
        # video.write(img)

        # frame = stream.read()
        key = cv2.waitKey(1) & 0xFF
        i = int(i)
        i = i +1
        if key == 27:  # exit
            break



if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, CHECKOUT_CLASSES as labelmap
    from ssd_network_architecture import build_ssd

    net = build_ssd('test', 300, 9) 
    net.load_state_dict(torch.load(args.weights))
    print(net)
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    cv2_demo(net.eval(), transform)

