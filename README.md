## Vehicle Object Detection
### Description
* As pedestrians taking the dog for a walk, escorting our kids to school, or marching to our workplace in the morning, we’ve all experienced unsafe, fast-moving vehicles operated by inattentive drivers that nearly mow us down.
* Many of us live in apartment complexes or housing neighborhoods where ignorant drivers disregard safety and zoom by, going way too fast.
We feel almost powerless. These drivers disregard speed limits, crosswalk areas, school zones, and “children at play” signs altogether. When there is a speed bump, they speed up almost as if they are trying to catch some air!
* Is there anything we can do?
* In most cases, the answer is unfortunately “no” — we have to look out for ourselves and our families by being careful as we walk in the neighborhoods we live in.
But what if we could catch these reckless neighborhood miscreants in action and provide video evidence of the which vehicle it is and the time of day to local authorities?
#### What is logo recognition? Logo recognition Model allow you to detect where on the internet your logo appears. Logo recognition is a must-have for any brands with a unique logo.
## Architecture
Google has released the object detection API for Tensorflow. It comes with several pre-implemented architectures with pre-trained weights on the COCO (Common Objects in Context) dataset, such as:
•	SSD (Single-Shot Multibox Detector) with MobileNets
•	SSD with Inception V2
•	R-FCN (Region-based Fully-Convolutional Networks) with Resnet 101
•	Faster RCNN (Region-based Convolutional Neural Networks) with Resnet 101
•	Faster RCNN with Inception Resnet v2
 
All these architectures are based on classification neural networks pre-trained on ImageNet. The design we choose to use for clothing item detection is Faster RCNN with Inception
 
Faster RCNN is a state-of-the-art model for deep learning-based object detection. RCNNs depend on region proposal algorithms to hypothesize object locations and then run a convolutional neural net on top of each of these region proposals, with a softmax classifier at the end. Faster RCNNs introduced RPNs (Region Proposal Networks) that share full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals.
For higher training and inference speed at the expense of accuracy, consider using SSD instead. SSD skips the region proposal step and finds every bounding box in every location of the image simultaneously with its classification. Because it does everything on one level, it is one of the fastest deep learning models for object detection and still performs quite comparably as the state-of-the-art.

### Data
We will be using the dataset gathered by scrapping from different internet sources that are openly available like Google Images, Bing Images and Pinterest. The size of the data is around 743Mb.

### Data Description

We have collected 15,677 images of different vehicles, which have a spread of 11 class labels assigned to them. Each class label is a different type of Vehicles, and we attempt to predict the vehicles through our CCTV. We resize the pictures to 256 × 256 pixels, and we perform both the model optimization and predictions on these downscaled images.

### Dataset Size:-

Total number of images in the custom fashion dataset 15,677
Total number of Training Images 12,542
Total number of Validation images 3135
Name of different categories: -
1.Aeroplane	7.Car
2.Auto	8.Scooty
3.Bicycle	9.Ship
4.Bike	10.Train
5.Boat	11.Truck
6.Bus	

Possible use cases in a different domain
## Data Augmentation

We can implement data augmentation by using the Augmentor Library. (https://github.com/mdbloice/Augmentor)
But we will not be performing this right now.
Install the library by-  pip install Augmentor
There are multiple functions that you can implement. Some of them are applied here. For more check https://augmentor.readthedocs.io/en/master/

      import Augmentor

# Initiating the Augmentor Pipeline
    p = Augmentor.Pipeline("C:\\Users\mr.india\Desktop\Images")

# Applying various type of Transformations and augmentation strategies
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        p.rotate90(probability=0.5)
        p.rotate270(probability=0.5)
        p.flip_left_right(probability=0.75)
        p.flip_top_bottom(probability=0.5)
        p.crop_random(probability=1, percentage_area=0.5)
        p.resize(probability=1.0, width=80, height=80)
        p.random_brightness(probability = 0.5, min_factor=0.4, max_factor=0.9)
        p.random_color(probability=0.5, min_factor=0.4, max_factor=0.9)
        p.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.4)
        p.random_distortion(probability=0.5, grid_width=7, grid_height=8, magnitude=9)
        p.random_erasing(probability=0.5, rectangle_area=0.4)
        p.zoom(probability=0.7, min_factor=1.1, max_factor=1.5)

#change the samples size according to requirements
p.sample(100)

### Data Annotation

In data annotation, we will be using Labelimg Tool
Download the Tool from the given Link:-
https://www.dropbox.com/s/kqoxr10l3rkstqd/windows_v1.8.0.zip?dl=1
 
Click on Open Dir and select the folder where your test images or train images are. Then start labelling all images in the dataset.
 
The selected is loaded in the Labelimg Interface.

 
## Click on Create RectBox.
 

Create the Bounding Box and write the Label name corresponding to the image. 

Select PascalVOC from the option. Click on the Save button and a corresponding XML file will be saved in the directory with the image.

### Train test validation

Total number of images in the custom fashion dataset 15,677
Total number of Training Images 12,542
Total number of Validation images 3135

### Training

We are using the Tensorflow Object Detection API.
https://github.com/tensorflow/models/tree/master/research/object_detection
 
    We will be using Tensorflow 1.14 for training the model. We will be training the model in virtual machines provided by Paperspace with P4000 GPU.
Download the complete github repository from 
https://github.com/tensorflow/models/tree/v1.13.0
 

After downloading it unzip the folder and rename the folder to models from models-master.
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
 
Download the repository and unzip it. Then rename the folder to objectdetection.
Visit the link to download the pretrained model.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 

#### This the official Tensorflow model zoo where all the pretrained models are available.
 
Click on faster_rcnn_inception_v2_coco, and then the model folder will be downloaded. Unzip the folder and keep it. Then rename it to “faster_rcnn_inception_v2_coco”.
Now we will be creating a folder structure.
We will be creating a folder in C drive called tensorflow_object_detection. Inside this folder we will be moving all our folders.
 
### Go to path   C:\tensorflow_object_detection\models\research. 
 
Create a new folder called images. This is where we will be keeping our own train and validation data. 
Create another new folder called training where we will be keeping our model config file and labelmap.pbtxt.
Now go to C:\tensorflow_object_detection\objectdetection\TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10-master
Copy two script files you need to train your custom model
•	Generate_tfrecords.py
•	xml_to_csv.py
  
And paste in  C:\tensorflow_object_detection\models\research folder.
C:\tensorflow_object_detection\models\research folder will be our prime folder from where we will be excuting all the terminal commands.
 
Go to  C:\tensorflow_object_detection\models\research\object_detection\samples\configs
 
Copy faster_rcnn_inception_v2_coco.config and paste in 
C:\tensorflow_object_detection\models\research\training
 
Go to  C:\tensorflow_object_detection\models\research\object_detection\legacy
 
Copy train.py and paste in C:\tensorflow_object_detection\models\research
Go to C:\tensorflow_object_detection\models\research\object_detection
Copy export_inference_graph.py and paste in 
C:\tensorflow_object_detection\models\research
 

For protobuff to py conversion
https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-win64.zip
Unzip the folder and rename to protoc and move to C drive.
 
Add protoc.exe to system environment variables path.
  
 
 
Run this command from C:\tensorflow_object_detection\models\research
"C:/protoc/bin/protoc" object_detection/protos/*.proto --python_out=.
Now the folder structure is ready.
Let’s create a new virtual environment using anaconda. I hope Anaconda is installed in your system. Open Anaconda prompt and type
conda create -n your_env_name python=3.6
your_env_name can be anything change to anything you prefer. I will be using vehicledetection.
 
Activate the vehicledetection environment.
 
Now you are inside your vehicledetection environment.
Now navigate to C:\tensorflow_object_detection\models\research
 
Install some dependent libraries
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
 
After the installation is successful.
 
 
python xml_to_csv.py
 
Now go to C:\tensorflow_object_detection\models\research\images  and verify.
 
test_labels.csv and train_labels.csv These two files are created.

Test tensorflow setup works in the object detection 
By opening the object_detection_tutorial.ipynb from object detection folder and executing all the cells.
Results should be
 
 

Now we need to generate tfrecord from csv files for training data and test data.
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

Create the labelmap.pbtxt
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt
 

Save in the folder C:\tensorflow_object_detection\models\research\training

Go to C:\tensorflow_object_detection\models\research\training
Open faster_rcnn_inception_v2_coco.config with text editor
 
Move the “faster_rcnn_inception_v2_coco” folder from C:\tensorflow_object_detection  to  C:\tensorflow_object_detection\models\research
 

 
All the red box cells contain the path that I am using. It may change based on your preferences.
Now let’s start the training process.
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
Training has started
You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the vehicledetection virtual environment, change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command:
(vehicledetection) C:\tensorflow1\models\research\object_detection>tensorboard --logdir=training

Training is completed after 8 hours in P4000 GPU.
Now we can see that checkpoint files are generated on the training folder.
Now we need to convert the last ckpt numbered file to the pb model we need.
For conversion
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory prediction_graph
In our case
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-200000 --output_directory prediction_graph
Now finally prediction_graph folder is generated.
 

Performance tuning
Possible tuning analytics
Comparison with benchmarks
Deployment
We will be deploying the model to the Pivotal Cloud Foundry platform. 
This is a workflow diagram for the prediction of diseases using the trained model.                  
                                                       

Now let’s see the Vehicle Object Detection project folder structure.
 

requirements.txt file consists of all the packages that you need to deploy the app in the cloud.

 

main.py is the entry point of our application, where the flask server starts. Here we will be decoding a base64 to an image, and then we will be doing predictions.

 
This is the obj.py file where the predictions take place based on the image we are giving input to the model.
 
manifest.yml:- This file contains the instance configuration, app name, and build pack language.

 
Procfile :- It contains the entry point of the app.

 
runtime.txt:- It contains the Python version number.

 
Visit the official website https://pivotal.io/platform.
 
Scroll Down to see the Start Trial Button



 
Click on the start trial button and the next interface will open. Then I will click on I’m ready to continue

 
Click on Download for Windows 64 bit, and then zip file will be downloaded. Keep it for future uses.
 
Now click on Let’s Keep Going
 
Click on Create Your Account
 
Fill Up your Details For registration
Do the email verification
Then login in again


 
After logging you will see this screen below and start your free trial.
Write any Company or which one you prefer
 
Enter your Mobile Number for Verification

 
Click on Pivotal Web Services

 
#### Give any Org name




 
- Now you are inside your Org and by default development space is created in your org. You can push your apps here.
- The cloud signup process is done and setup is ready for us to push the app.

- Previously you have downloaded the CLI.zip file. Unzip the file and install the .exe file with admin rights.
- After successful installation, you can verify by opening your CMD and type cf. 
- Then you will get a screen which is shown below
 

#### If you see this screen in your CMD the installation is successful.

Now type the command to login via cf-cli
cf login -a https://api.run.pivotal.io
Next, enter your email and password. Now you are ready to push your app.
Now let’s go to the app which we have built.
 

Navigate to the project folder after downloading from the given below link:-

Then write cf push in the terminal.
 

After the app is successfully deployed in the cloud you will see the below screen with the route.


 
#### Finally, the app is pushed the cloud.

- Now let’s verify the API with some image if the API is working correctly and giving response.

- So our Api is : - https://vehiclemultidetection-appreciative-dingo.cfapps.io/
- For prediction we have added /predict route which will give us the results.
- We will use Postman for verification. 
 
We will use this image for verification purpose which is brand logo of Aeroplane.

Let’s encode the image to base64 from https://base64.guru/converter/encode/image (Use any converter)

##### Let’s open Postman

write postman work area like {"image":"past over here your base64 string"}

convert image into base64 string and then put the converted image into your postman and then send POST Request to seee the output
 
### Your output

className : Aeroplane
conifidence : 0.81095...
xMax :
xMin :
yMax :
yMin :

### Conclusion

* Building a vehicle detection model from scratch is overwhelming, especially when no database about users is yet accessible. It is not uncommon to default models behavior to dummy recommendations (like most trendy clothes) when tackling the cold-start problem. Therefore, being able to isolate clothing items from an outfit to perform more advanced queries at an early stage of development of a recommender engine proves itself irreplaceable.
* 
* Another major use case of the clothes detector is to be able to show information about a specific clothing item part of an outfit when the user hovers or clicks on it in an app or website. The same trained network can be used for this purpose.
Training object detection networks on the cloud, such as Faster RCNN takes around 8hours of time.
Credits To:-
https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/
