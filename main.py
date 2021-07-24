#@title <b><font color="red" size="+3">←</font><font color="black" size="+3"> Clone Git repository and install all requirements</font></b>
#%tensorflow_version 1.x

import os,glob,shutil
import cv2
import math
import pickle
import imageio
import warnings
import PIL.Image
import numpy as np
from PIL import Image
import tensorflow as tf
from random import randrange
import matplotlib.pyplot as plt

#%matplotlib inline
warnings.filterwarnings("ignore")




def generate_final_image(latent_vector, direction, coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    img.thumbnail(size, PIL.Image.ANTIALIAS)
    #img.save("face.png")
    # if download_image == True: files.download("face.png")
    return img

def plot_three_images(img_result, fs = 10, imgA=None,imgB=None):
  f, axarr = plt.subplots(1,3, figsize=(fs,fs))
  if imgA is None:
      imgA = Image.open('aligned_images/A1_01.png')
  if imgB is None:
      imgB = Image.open('aligned_images/C0_01.png')
  axarr[0].imshow(imgA)
  axarr[0].title.set_text("father")
  axarr[1].imshow(img_result)
  axarr[1].title.set_text("child")
  axarr[2].imshow(imgB)
  axarr[2].title.set_text("mother")
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
  # plt.savefig('result_01')
  # plt.show()


import config
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator

age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
horizontal_direction = np.load('ffhq_dataset/latent_directions/angle_horizontal.npy')
vertical_direction = np.load('ffhq_dataset/latent_directions/angle_vertical.npy')
eyes_open_direction = np.load('ffhq_dataset/latent_directions/eyes_open.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')







tflib.init_tf()
URL_FFHQ = "karras2019stylegan-ffhq-1024x1024.pkl"
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
model_scale = int(2*(math.log(1024,2)-1))





person_age = 28.0
intensity = -((person_age/5)-6)
imgA_list = glob.glob('aligned_images/A_*.png')
imgB_list = glob.glob('aligned_images/B_*.png')

for imgA_path in imgA_list:
    for imgB_path in imgB_list:
        buff = []



        A_id = os.path.basename(imgA_path).split('A_')[-1].split(('_01.png'))[0]
        B_id = os.path.basename(imgB_path).split('B_')[-1].split(('_01.png'))[0]
        saved_path = 'result/'+A_id+'-'+B_id
        saved_merge_path = saved_path+'/merge'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        if not os.path.exists(saved_merge_path):
            os.makedirs(saved_merge_path)
        shutil.copy2(imgA_path,saved_path)
        shutil.copy2(imgB_path, saved_path)

        imgA = Image.open(imgA_path)
        imgB = Image.open(imgB_path)

        first_face = np.load('latent_representations/'+os.path.basename(imgA_path).split('.png')[0]+'.npy')
        second_face = np.load('latent_representations/'+os.path.basename(imgB_path).split('.png')[0]+'.npy')

        for weight in np.linspace(0.01,0.5,30):

            hybrid_face = ((1-weight)*first_face)+(weight*second_face)
            resolution = "512" #@param [256, 512, 1024]
            size = int(resolution), int(resolution)

            face = generate_final_image(hybrid_face, age_direction, intensity)
            face.save(saved_merge_path+ '/merge_' + str(weight) + '.png')

            img = np.asarray(face)


        #     plot_three_images(face, fs=15,imgA=imgA,imgB=imgB)
        #     plt.savefig(str(round(person_age,1))+'_'+str(weight)+'.png')
        #     plt.clf()
        #     #face.save(str(person_age)+'_'+str(weight)+'.png')
        #     #cv2.imwrite(str(weight)+'.png',np.array(face))
            buff.append(img)
        face.save(saved_path + '/merge_' + str(weight) + '.png')
        gif = imageio.mimsave(saved_path+'/merge.gif', buff, 'GIF', duration=0.2)
