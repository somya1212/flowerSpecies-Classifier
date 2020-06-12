# flowerSpecies-Classifier
AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, we would use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice I would train this classifier, then export it for use in our application. I'll be using this dataset(http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
## Following arguments are mandatory or optional for train.py

- 'data_dir'. 'Provide data directory. Mandatory argument', type = str
- '--save_dir'. 'Provide saving directory. Optional argument', type = str
- '--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str
- '--lrn'. 'Learning rate, default value 0.001', type = float
- '--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
-'--epochs'. 'Number of epochs', type = int
- '--GPU'. "Option to use GPU", type = str

## Following arguments are mandatory or optional for predict.py

- 'image_dir'. 'Provide path to image. Mandatory argument', type = str
- 'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
- '--top_k'. 'Top K most likely classes. Optional', type = int
- '--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
- '--GPU'. "Option to use GPU. Optional", type = str
