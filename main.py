#@title <b><font color="red" size="+3">←</font><font color="black" size="+3"> Clone Git repository and install all requirements</font></b>
#%tensorflow_version 1.x

import os
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

def get_watermarked(pil_image: Image) -> Image:
  try:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
    pct = 0.08
    full_watermark = cv2.imread('media/logo.png', cv2.IMREAD_UNCHANGED)
    (fwH, fwW) = full_watermark.shape[:2]
    wH = int(pct * h*2)
    wW = int((wH * fwW) / fwH*0.1)
    watermark = cv2.resize(full_watermark, (wH, wW), interpolation=cv2.INTER_AREA)
    overlay = np.zeros((h, w, 4), dtype="uint8")
    # (wH, wW) = watermark.shape[:2]
    # overlay[h - wH - 10 : h - 10, 10 : 10 + wW] = watermark
    output = image.copy()
    #cv2.addWeighted(overlay, 0.5, output, 1.0, 0, output)
    rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
  except: return pil_image

def generate_final_images(latent_vector, direction, coeffs, i):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if size[0] >= 512: img = get_watermarked(img)
    img_path = "for_animation/" + str(i) + ".png"
    img.thumbnail(animation_size, PIL.Image.ANTIALIAS)
    img.save(img_path)
    face_img.append(imageio.imread(img_path))
    return img



def generate_final_image(latent_vector, direction, coeffs):
    new_latent_vector = latent_vector.copy()
    new_latent_vector[:8] = (latent_vector + coeffs*direction)[:8]
    new_latent_vector = new_latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(new_latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    if size[0] >= 512: img = get_watermarked(img)
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


if len(os.listdir('generated_images')) >= 2:
  first_face = np.load('latent_representations/A1_01.npy')
  second_face = np.load('latent_representations/C0_01.npy')
  print("Generation of latent representation is complete! Now comes the fun part.")
else: raise ValueError('Something wrong. It may be impossible to read the face in the photos. Upload other photos and try again.')



imgA = Image.open('aligned_images/A1_01.png')
imgB = Image.open('aligned_images/C0_01.png')

for weight in [0.2,0.4,0.6,0.8]:
    buff = []
#weight = 0.7
    for person_age in np.linspace(1,15,5):

        hybrid_face = ((1-weight)*first_face)+(weight*second_face)

        #person_age = 30
        intensity = -((person_age/5)-6)


        #@markdown **Resolution of the downloaded image:**
        resolution = "512" #@param [256, 512, 1024]
        size = int(resolution), int(resolution)

        face = generate_final_image(hybrid_face, age_direction, intensity)
        img = np.asarray(face)
        # cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2BGR)
        #plt.imsave()
        buff.append(img)
        plot_three_images(face, fs=15,imgA=imgA,imgB=imgB)
        plt.savefig(str(round(person_age,1))+'_'+str(weight)+'.png')
        plt.clf()
        #face.save(str(person_age)+'_'+str(weight)+'.png')
        #cv2.imwrite(str(weight)+'.png',np.array(face))
    gif = imageio.mimsave('face_'+str(weight)+'.gif', buff, 'GIF', duration=0.2)
