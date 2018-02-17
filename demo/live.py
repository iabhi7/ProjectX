from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import os
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/media/predible/sdd/Abhishek/ssd.pytorch/weights/ssd300_0712_30000.pth',
                    type=str, help='Trained state_dict file path')
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
            while detections[0, i, j, 0] >= 0.4:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                int(pt[3])), COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), FONT,
                            2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    
    videoFilePath = "/media/predible/sdd/Abhishek/ssd.pytorch/data/dataset/video_test.avi"
    print('home/abhishek/data/VOCdevkit/dataset/video_test.avi')
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
        save = os.path.join('/home/abhishek/images', i + '.png')
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

# img1 = cv2.imread('1.jpg')
# img2 = cv2.imread('2.jpg')
# img3 = cv2.imread('3.jpg')

# height , width , layers =  img1.shape

# video = cv2.VideoWriter('video.avi',-1,1,(width,height))

# video.write(img1)
# video.write(img2)
# video.write(img3)

    # stream = WebcamVideoStream(src=0).start()  # default camera
    # time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    # while True:
    #     # grab next frame
    #     frame = stream.read()
    #     key = cv2.waitKey(1) & 0xFF

    #     # update FPS counter
    #     fps.update()
    #     frame = predict(frame)

    #     # keybindings for display
    #     if key == ord('p'):  # pause
    #         while True:
    #             key2 = cv2.waitKey(1) or 0xff
    #             cv2.imshow('frame', frame)
    #             if key2 == ord('p'):  # resume
    #                 break
    #     cv2.imshow('frame', frame)
    #     if key == 27:  # exit
    #         break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 8)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    print(net)
    transform = BaseTransform(net.size, (132/256.0, 144/256.0, 147/256.0))

#     x = cv2.resize(image, (300, 300)).astype(np.float32)
# # x -= (.4347146267361, 144.68103624131945, 147.3058550347222)
# print(cv2.mean(image))
# # x -= (145.3834450954861, 158.69114366319445, 157.93541666666667)
# # x -= (104.0, 117.0, 123.0)
# x = x.astype(np.float32)
# x = x[:, :, ::-1].copy()
# # plt.imshow(x)
# x = torch.from_numpy(x).permute(2, 0, 1)
# xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable

    # fps = FPS().start()
    # videoFilePath = "/home/abhishek/data/VOCdevkit/dataset/vieo_test.avi"
    # videoFile = cv2.VideoCapture(videoFilePath)
    # while True:
    #     ret, image = videoFile.read()
    #     demo_video(net.eval(),image)
    # stop the timer and display FPS information
    cv2_demo(net.eval(), transform)
    # fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()
    stream.stop()
