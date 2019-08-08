## Cloud Classification

# Requirements:

 * Python 3.6 
	* alread installed on all environments (linux,unix etc)
	* check for version: 
	>> python.__version__
 * Tensorflow 
	>> pip install tensorflow-gpu or pip install tensorflow
 * NumPy
	>> pip install numpy
 * OpenCV2
	>> pip install python-opencv



# Sample class_index:
classes = {'hareteru':0 , 'ichibu_kumori':1 , 'kumori':2 , 'takai_kumo':3  } 

Therefore class_index will take one of:
class_indexes = [0,1,2,3]

# Note:
* "class_index" must start from "0".
* Number of classes is going to be len(class_indexes)
Depending on number of classes, we have to define following flag as well:
"--num_classes" which is going to be equal len(class_indexes)

#For Predictions

>>  python predict.py --test_img_dir /path/to/test_images --model_path /path/to/Cloud_Model/ckpt/model_epoch100.ckpt --pred_out_dir /path/to/Cloud_Model/pred_out


## Train a Model

1.Open two different files  and name them as below: 

File 1: train.txt
File 2: test.txt

2. Write image paths and corresponding class of that image in each line.

# Sample content of each file:

train.txt:
/absolute/path/to/image1.jpg class_index
/absolute/path/to/image2.jpg class_index
/absolute/path/to/image3.jpg class_index
/absolute/path/to/image4.jpg class_index
/absolute/path/to/image5.jpg class_index
/absolute/path/to/image6.jpg class_index
...

test.txt:
/absolute/path/to/image1.jpg class_index
/absolute/path/to/image2.jpg class_index
/absolute/path/to/image3.jpg class_index
/absolute/path/to/image4.jpg class_index
/absolute/path/to/image5.jpg class_index
/absolute/path/to/image6.jpg class_index
...

## Checking Image Paths

Make sure file paths are true by simply checking with os.path.exist:

with open("../train.txt","r") as f:
	lines = f.readlines()
	for each_line in lines:
		if os.path.exist(each_line.split()[0]):
			print("File path is true")
		else:
			print("please check if file path is updated")

* each_line.split()[0] will split image path and class_index and returns image path only. 
* Same process can be applied for "test.txt".


## Train by Changing Parameters:

>> python finetune_res.py --training_file /path/to/Cloud_Model/data/train.txt --val_file /path/to/Cloud_Model/data/val.txt --num_epochs 50 --num_classes 11 --batch_size 32 --learning_rate 0.001 --resnet_depth 50


Reference papers:

 * [ResNet](https://arxiv.org/pdf/1512.03385.pdf)






