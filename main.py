from audio_tools import *
from video_tools import *

timeSlice = 1 #seconds

sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-small.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-medium.mov"
# sample = "https://s3-us-west-2.amazonaws.com/e7mac.com/BreakMeMadeira-full.mov"

audioSlice("BreakMeMadeira31.mov", timeSlice)
# videoSlice("BreakMeMadeira31.mov", timeSlice, 10, 0.1)
