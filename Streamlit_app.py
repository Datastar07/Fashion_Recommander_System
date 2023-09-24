#Importing all the python libraries which is used in this project.
import os
import pickle
import tensorflow
import numpy as np
from PIL import Image
import streamlit as st
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

#Importing the feature_list and its filenmaes pkl file,
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

#Deep learning model,
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#crete the webpage for deployment the recommandation system,
st.title('Fashion Recommender System')

#This function take images from the upload file and store them into the backend,
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

#This function extract features from the uploaded images,
def feature_extraction(img_path,model):
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

#This function recommandate the all 5 vector related to the uploaded image,
def recommend(features,feature_list):
    #KNN algorithm for finding the nearest features from the our image. 
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)

        # recommendention
        indices = recommend(features,feature_list)

        # showing the recommandation
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
            
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occured in file upload")