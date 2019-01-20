import glob
import numpy as np
from shutil import copyfile
from natsort import natsorted

p='/home/akshita/salman/ZSD_Polar/Dataset/correct/video4_s'
files=glob.glob(p+'/*.jpg')
tar='/home/akshita/salman/ZSD_Polar/Dataset/correct/video4_s_mod'
print(len(files))
files=natsorted(files)
for i in range(len(files)):
	src = files[i]
	filename='filename%03d.jpg'%int(i+1)
	#print(filename)
	copyfile(src,tar+'/'+filename)
