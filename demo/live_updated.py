from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import os
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/Users/abhishek/Downloads/ssd_new_mean_0712_5000.pth',
                    type=str, help='Trained state_dict file path')
# parser.add_argument('--weights', default='/Users/abhishek/Downloads/ssd300_0712_10000.pth',
#                     type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net):
    # print(net)
    def predict(frame):
        # height, width = frame.shape[:2]
        # x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x = cv2.resize(frame, (300, 300)).astype(np.float32)
        x -= (124, 128, 125)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()

        x = torch.from_numpy(x).permute(2, 0, 1)


        xx = Variable(x.unsqueeze(0))   
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)


        # x = Variable(x.unsqueeze(0))
        # y = net(x)  # forward pass

        detections = y.data
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        top_k=10
        # scale = torch.Tensor([width, height, width, height])
        scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.2:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    
    videoFilePath = "/Users/abhishek/Documents/data/MVI_1499.MP4"
    print('home/abhishek/data/VOCdevkit/dataset/video_test.avi')
    videoFile = cv2.VideoCapture(videoFilePath)
    array = []
    i = 0
    while True:
        ret, image = videoFile.read()
        # print(ret, image)
        # print("printing")
        i = str(i)
        output = predict(image)
        print("---------------", i)
        save = os.path.join('/Users/abhishek/images', i + '.png')
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

        # # update FPS counter
        # fps.update()
        # frame = predict(frame)

        # keybindings for display
        # if key == ord('p'):  # pause
        #     while True:
        #         key2 = cv2.waitKey(1) or 0xff
        #         cv2.imshow('frame', frame)
        #         if key2 == ord('p'):  # resume
        #             break
        # cv2.imshow('frame', image)
        if key == 27:  # exit
            break
    # print(len(array))


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 13)    # initialize SSD
    net.load_weights('/Users/abhishek/Downloads/ssd_new_mean_0712_5000.pth')
    print(net)
    # transform = BaseTransform(net.size, (132/256.0, 144/256.0, 147/256.0))


    cv2_demo(net.eval())
    # fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
