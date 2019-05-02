# from glob import glob
# import os

# # p = 'Training-Normal-Videos-Part-1/*.mp4'
# p = 'Training-Normal-Videos-Part-2/*.mp4'
# p2 = 'frames2'
# files= glob(p)
# #os.mkdir(p2)
# for file_ in files:
#     filename = os.path.split(file_)[-1]#.split('.')[0]
#     foldname = os.path.split(file_)[-1].split('.')[0]
#     os.makedirs(os.path.join(p2,foldname))
#     #print(filename)
#     os.system('ffmpeg -i Training-Normal-Videos-Part-2/{} frames2/{}/filename%03d.jpg'.format(filename,foldname))

from glob import glob
import os

# p = 'Training-Normal-Videos-Part-1/*.mp4'
src = 'videos/'
p = src +'/*.mp4'
tar = 'frames/'
files= glob(p)
#os.mkdir(p2)
for file_ in files:
    filename = os.path.split(file_)[-1]#.split('.')[0]
    foldname = os.path.split(file_)[-1].split('.')[0]
    os.makedirs(os.path.join(tar,foldname))
    #print(filename)
    os.system('ffmpeg -i {}/{} {}/{}/filename%03d.jpg'.format(src,filename,tar,foldname))