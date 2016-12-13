#coding=utf-8
import sys
import os
import numpy as np
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib

wPath='/media/a00001/E/age/IMDB/train/'
imdb = loadmat('/media/a00001/E/age/IMDB/imdb_meta/imdb/imdb.mat')['imdb']
dob=imdb['dob'][0][0][0]
photo_taken=imdb['photo_taken'][0][0][0]
full_path=imdb['full_path'][0][0][0]
gender=imdb['gender'][0][0][0]
name=imdb['name'][0][0][0]
face_location=imdb['face_location'][0][0][0]
face_score=imdb['face_score'][0][0][0]
second_face_score=imdb['second_face_score'][0][0][0]
celeb_names=imdb['celeb_names'][0][0][0]
celeb_id=imdb['celeb_id'][0][0][0]
info = open('/media/a00001/E/age/IMDB/InfoFile.txt', 'w')
def Score(s):
    if('nan'==str(s)):
        return 0
    else:
        return s
for i in range(len(dob)):
    if(('nan'!=str(gender[i]))and (os.path.exists(wPath+str(full_path[i][0].encode("utf-8"))))):
        print(wPath+full_path[i][0].encode("utf-8")+str(' yes'))
        info.write(full_path[i][0].encode("utf-8")+' '+str(gender[i]))
        info.write(' '+str(dob[i]))
        info.write(' ' + str(photo_taken[i]))
        info.write(' ' + str(Score(face_score[i])))
        info.write(' ' + str(Score(second_face_score[i])))
        info.write(' ' + name[i][0].encode("utf-8"))
        info.write('\n')
    else:
        print(wPath + full_path[i][0].encode("utf-8") + str(' no'))