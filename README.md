
# Fashion Recommender system

## Description ✍️
- **Recommandation **
With an increase in the standard of living, peoples' attention gradually moved towards fashion that is concerned to be a popular aesthetic expression. Humans are inevitably drawn towards something that is visually more attractive. This tendency of humans has led to the development of the fashion industry over the course of time. However, given too many options of garments on the e-commerce websites, has presented new challenges to the customers in identifying their correct outfit. Thus, in this project, we proposed a personalized Fashion Recommender system that generates recommendations for the user based on an input given. Unlike the conventional systems that rely on the user's previous purchases and history, this project aims at using an image of a product given as input by the user to generate recommendations since many-a-time people see something that they are interested in and tend to look for products that are similar to that. We use neural networks to process the images from Fashion Product Images Dataset and the Nearest neighbour backed recommender to generate the final recommendations.

## Simple App UI Demo 🖼️
![gif](https://github.com/Datastar07/Fashion_Recommander_System/blob/main/Demo/Fashion_Demo_GIF.gif)

## Related work 💼

In the online internet era, the idea of Recommendation technology was initially introduced in the mid-90s. Proposed CRESA that combined visual features, textual attributes and visual attention of 
the user to build the clothes profile and generate recommendations. Utilized fashion magazines 
photographs to generate recommendations. Multiple features from the images were extracted to learn 
the contents like fabric, collar, sleeves, etc., to produce recommendations. In order to meet the 
diverse needs of different users, an intelligent Fashion recommender system is studied based on 
the principles of fashion and aesthetics. To generate garment recommendations, customer ratings and 
clothing were utilized in The history of clothes and accessories, weather conditions were 
considered in to generate recommendations.

##  Proposed methodology 👨‍💻

In this project, we propose a model that uses Convolutional Neural Network,this model name is RESNET-50 model which is pretrained model  and the Nearest neighbour backed recommender. As shown in the figure Initially, the neural networks are trained and then an inventory is selected for generating recommendations and a database is created for the items in inventory. The nearest neighbour’s algorithm is used to find the most relevant products based on the input image and recommendations are generated.

![Alt text](https://github.com/Datastar07/Fashion_Recommander_System/blob/main/Demo/work-model.png)

## Training the neural networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
 shows the ResNet50 architecture.

![Alt text](https://github.com/Datastar07/Fashion_Recommander_System/blob/main/Demo/resnet.png)

## Getting the inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![Alt text](https://github.com/Datastar07/Fashion_Recommander_System/blob/main/Demo/inventry.png)



## Experiment and results

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. 
Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441
garment images. The networks are trained and validated on the dataset taken. The training results 
show a great accuracy of the model with low error, loss and good f-score.


## Installation ⬇️

Use pip to install the requirements.

~~~bash
pip install -r requirements.txt
~~~

## Usage
First of all you need to download the datatset from this link and put all the [images](https://github.com/Datastar07/Fashion_Recommander_System/tree/main/images) images into this images folder.
 
## Dataset Link:
[Kaggle Dataset Big size 25 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)


For extracting the feature from the image you need to run this line:
```bash
python Extracting_feature.py
```

After generating the features.pkl file you need to run below line for showing the result on the web app,
To run the web server, simply execute streamlit with the main recommender app:

```bash
streamlit run Streamlit_app.py
```
##Note: ⚠️
In which when we are running the python Extracting_feature.py file it takes some time because our dataset have images arround 43000 so for that reason it took some time.


## Conclusion

In this project, we have presented a novel framework for fashion recommendation that is driven by data, 
visually related and simple effective recommendation systems for generating fashion product images. 
The proposed approach uses a two-stage phase. Initially, our proposed approach extracts the features 
of the image using CNN classifier ie., for instance allowing the customers to upload any random 
fashion image from any E-commerce website and later generating similar images to the uploaded image 
based on the features and texture of the input image. It is imperative that such research goes forward 
to facilitate greater recommendation accuracy and improve the overall experience of fashion 
exploration for direct and indirect consumers alike.
