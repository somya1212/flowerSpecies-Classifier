# flowerSpecies-Classifier
AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, we would use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice I would train this classifier, then export it for use in our application.


## Dataset:

I'll be using this dataset(http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

## Workflow of the Project:

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

## Software and Libraries
This project uses the following software and Python libraries:

- Python
- NumPy
- pandas
- Matplotlib
- PyTorch
You also need to have additional software installed to run and execute a Jupyter Notebook.

If you do not have Python installed, I highly recommend installing the Anaconda distribution of Python, which already has the above packages and more included.

## Run
In a terminal or command window, navigate to the top-level project directory Image Classifier Project/ (that contains this README) and run one of the following commands:

`ipython notebook Image Classifier Project.ipynb`

or

`jupyter notebook Image Classifier Project.ipynb`

## For the command line app

Train a new network on a data set with train.py
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

### Options:
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory

* Choose architecture: python train.py data_dir --arch "vgg13"

* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 3

* Use GPU for training: python train.py data_dir --gpu

* Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

* Basic usage: python predict.py /path/to/image checkpoint

### Options:
* Return top KK most likely classes: python predict.py input checkpoint --top_k 3

* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json

* Use GPU for inference: python predict.py input checkpoint --gpu

* This will open the iPython Notebook software and project file in your browser.


### Following arguments are mandatory or optional for train.py

- 'data_dir'. 'Provide data directory. Mandatory argument', type = str
- '--save_dir'. 'Provide saving directory. Optional argument', type = str
- '--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str
- '--lrn'. 'Learning rate, default value 0.001', type = float
- '--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
-'--epochs'. 'Number of epochs', type = int
- '--GPU'. "Option to use GPU", type = str

### Following arguments are mandatory or optional for predict.py

- 'image_dir'. 'Provide path to image. Mandatory argument', type = str
- 'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
- '--top_k'. 'Top K most likely classes. Optional', type = int
- '--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
- '--GPU'. "Option to use GPU. Optional", type = str

## Contributions :
You are welcome to suggest any changes or add contributions.

## License:

This project is covered under the MIT License.
