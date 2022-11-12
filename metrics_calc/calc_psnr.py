# Code taken from https://github.com/chia56028/PSNR-Calculator-for-Images

import math
import cv2
import csv
import os
import glob
from datetime import datetime

def read_bmp(image_name):
	return cv2.imread(image_name+'.jpg')

def calculate_psnr(s,r):

	height, width, channel = s.shape
	size = height*width

	sb,sg,sr = cv2.split(s)
	rb,rg,rr = cv2.split(r)

	mseb = ((sb-rb)**2).sum()
	mseg = ((sg-rg)**2).sum()
	mser = ((sr-rr)**2).sum()

	MSE = (mseb+mseg+mser)/(3*size)
	psnr = 10*math.log10(255**2/MSE)
	return round(psnr,2)

# def write_csv(n,data):
# 	with open('PSNR-result/'+n+'.csv', 'w', newline='') as myfile:
# 		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
# 		wr.writerow(data)

# iter = '2299'
# for i in range(1,2):
#     name = iter + '_' + str(i+1)
#     print("Creating CSV of PSNR-result",i+1,"...",sep="")
#     write_csv(str(i+1),[calculate_psnr(name)])

# os.system("pause")

files = sorted(glob.glob("./recovery/*.jpg"))

with open('psnr_result.txt', 'a') as myfile:
    myfile.write('PSNR Result ' + str(datetime.now()) + '\n')

for each_file in files:
    # Assuming single channel images are read. For RGB image, uncomment the following commented lines
    img1 = cv2.imread(each_file)
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("./source/source.jpg")
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    psnr = calculate_psnr(img1, img2)

    with open('psnr_result.txt', 'a') as myfile:
        myfile.write(each_file.split('/')[-1] + '------> PSNR ')
        myfile.write(str(psnr) + '\n')

with open('psnr_result.txt', 'a') as myfile:
    myfile.write('\n')
    myfile.close()