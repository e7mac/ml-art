def videoSlice(filename, sliceDuration, fps, sizeFraction):
    import cv2
    print(cv2.__version__)
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
        outputFilename = "output_video/slice_%d_frame%d.jpg" % (sliceNum, count)
        cv2.imwrite(outputFilename, image)     # save frame as JPEG file
        success,image = vidcap.read()
        if success == True:
            image = cv2.resize(image, (0,0), fx=sizeFraction, fy=sizeFraction)
        print('Read a new frame: ' + str(count))
        count += 1
        if count%countPerSlice==0:
            sliceNum = sliceNum + 1

