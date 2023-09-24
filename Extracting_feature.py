#Importing all the python libraries which is used in this project.
import os
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from tqdm import tqdm
import pickle

#create the Deep learning model,
model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#This function shows our RESNET-50 model parameters.
print(model.summary())

#create the function which extract the features from the all images,
def extract_features(img_path,model):
    #Load the image.
    img = tensorflow.keras.utils.load_img(img_path, target_size=(224, 224))

    #Convert the image into the array.
    img_array = tensorflow.keras.utils.img_to_array(img)

    #Expand the image the array.
    expanded_img_array = np.expand_dims(img_array, axis=0)

    #preprocess the expanded image array.
    preprocessed_img = preprocess_input(expanded_img_array)

    #predict the feature from the image using our deep learning model.
    result = model.predict(preprocessed_img).flatten()

    #Normalized the result.
    normalized_result = result / norm(result)

    #Return the normalized results.
    return normalized_result

#Append all the images path in this list.
filenames=[]

for files in os.listdir("images"):
    filenames.append(os.path.join("images",files))

feature_list=[]

for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

#pickle the all the files which contains the features from it.
pickle.dump(feature_list,open("embeddings.pkl","wb"))
pickle.dump(filenames,open("filenames.pkl","wb"))