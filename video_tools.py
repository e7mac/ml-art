import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class VideoDataSet():
    """A class to store video data set for AI processing"""
    imgs = []

    def __init__(self, filename):
        self.filename = filename
        self.imgs = self.videoRead(filename)
        self.setDtype()

    def setDtype(self):
        self.imgs = self.imgs / 255.
        self.imgs = self.imgs.astype(np.float32, copy=False)


    def videoRead(self, filename):
        cap = cv2.VideoCapture(filename)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        cap.release()
        return buf

    def videoSlice(self, filename, sliceDuration, fps, sizeFraction):    
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        image = cv2.resize(image, (0,0), fx=sizeFraction, fy=sizeFraction)
        count = 0
        sliceNum = 0
        success = True
        countPerSlice = int(sliceDuration * fps)
        desired_fps = fps
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000.0 / desired_fps))
            outputFilename = "output/slice_%d_frame%d.jpg" % (sliceNum, count)
            cv2.imwrite(outputFilename, image)     # save frame as JPEG file
            success,image = vidcap.read()
            if success == True:
                image = cv2.resize(image, (0,0), fx=sizeFraction, fy=sizeFraction)
            print('Read a new frame: ' + str(count))
            count += 1
            if count%countPerSlice==0:
                sliceNum = sliceNum + 1

    def extractAudio(self):
        os.system('ffmpeg -i ' + self.filename + ' ' + self.filename + '.wav')
        # clip = mp.VideoFileClip(filename)
        # clip.audio.write_audiofile(filename + ".wav")

    def displayImg(self, index):
        plt.imshow(self.imgs[index])
        plt.show()

    def audioFilename(self):
        return self.filename + ".wav"
