import imageio

images = []
path = "/home/hpaat/FF/xray/"
filed = "" # "file_brain_AXT2_200_2000019.h5|langevin|slide_idx_0_R=4_"

filenames = []
for num in range(1,6,1):
    filenames.append(str(num))

for filename in filenames:
    #images.append(imageio.imread(path + filed + str(filename) + ".jpg"))
    #import pdb; pdb.set_trace() 
    images.append(imageio.imread(path + str(filename) + ".jpg"))
imageio.mimsave(path + 'animated2.gif', images, duration=1)