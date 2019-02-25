import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class VideoDataSet():
    """A class to store video data set for AI processing"""
    imgs = []
    frameDimension = 200

    def __init__(self, filename):
        self.filename = filename
        self.imgs = self.videoRead(filename)
    #     self.setDtype()

    # def setDtype(self):
    #     self.imgs = self.imgs / 255.
    #     self.imgs = self.imgs.astype(np.float32, copy=False)

    def mean(self):
        return np.mean(self.imgs, axis=0)

    def std(self):
        return np.std(self.imgs, axis=0) + 1e-10

    def preprocess(self, img):
        return (img - self.mean()) / self.std()

    def deprocess(self, img):
        return img * self.std() + self.mean()

    def n_features(self):
        """This is for linear flattening"""
        shape = self.imgs.shape
        return shape[1] * shape[2] * shape[3]

    def videoRead(self, filename):
        cap = cv2.VideoCapture(filename)
        self.frameRate = int(cap.get(cv2.CAP_PROP_FPS))
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        minDim = min(self.frameWidth, self.frameHeight)
        dim = (minDim, minDim)
        buf = np.empty((self.frameCount, self.frameDimension, self.frameDimension, 3), np.dtype('float32'))
        frame = np.empty((1, self.frameHeight, self.frameWidth, 3), np.dtype('float32'))
        fc = 0
        ret = True
        while (fc < self.frameCount  and ret):
            ret, frame = cap.read()
            buf[fc] = cv2.resize(frame, (self.frameDimension, self.frameDimension))
            buf[fc] = buf[fc] / 255.0
            fc += 1
        cap.release()
        buf = buf[...,::-1] # opencv gets BGR.. convert to RGB
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
        os.system('ffmpeg -n -i ' + self.filename + ' ' + self.filename + '.wav')
        # clip = mp.VideoFileClip(filename)
        # clip.audio.write_audiofile(filename + ".wav")

    def displayImg(self, index):
        plt.imshow(self.imgs[index])
        plt.show()

    def audioFilename(self):
        return self.filename + ".wav"
