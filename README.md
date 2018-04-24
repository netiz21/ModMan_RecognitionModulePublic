# ModMan object detection and pose estimation module

This is an object detection and pose estimation module in Modular Manipulation Project.

The object detection part is mainly based on the [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) that refers to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
Pose estimation part is composed of SURF and PnPSolver of [OpenCV](https://opencv.org/)

Additional parts such as various inputs (web camera, realsense, image and video) and communication modules are added to the main code. 

### Requirements: hardware

For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)


### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

 In python 2.7
 ```
    sudo apt-get install python-pip python-dev
    pip install tensorflow-gpu
 ```
 In python 3.n
 ```Shell
    sudo apt-get install python3-pip python3-dev
    pip3 install tensorflow-gpu
 ```
 
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`


### Installation

1. install Ubuntu, NVIDIA driver, CUDA, cudnn
https://yochin47.blogspot.com/b/post-preview?token=jRWq4mIBAAA.VK6RHTAdt_xdkHa1MfMMbAZtZ5o7lNvE6YZ_cBAIkiMWnQ2U0rtDn21vvyV-s1suOevKFKz3ScjRBU8LcmAefQ.GVl7Oypj1ycMYh32u0NF2g&postId=6045618232993691092&type=POST

2. Anaconda installation
import ModMan conda env.

3. Download Faster-RCNN_TF based ModMan recognition module
git clone https://github.com/yochin/smallcorgi_Faster-RCNN_TF_yochin.git

4. compile
source activate ModMan
cd FRCN/lib
make

error: roi_pooling_op.cu.o: No such file or directory
add  to your path
export PATH="/usr/local/cuda-8.0/bin:$PATH"

5. download trained model
move DBv1 folder to FRCN/yochin_tools/PoseEst
move models folder to FRCN

6. set path & run



* If you meet a erro message like 'import module error ###', then install that module using below command.
source activate ModMan

pip install pyyaml
conda install -c auto easydict
conda install -c auto scipy
conda install -c https://conda.anaconda.org/menpo opencv
conda install -c conda-forge hdf5 
conda install -c auto matplotlib
conda install -c anaconda h5py


1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/yochin/smallcorgi_Faster-RCNN_TF_yochin.git
  ```
  or download and unzip the zipped file.
  
  The unzipped folder == FRCN_ROOT

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```
    
    If you meet an error "No moduel named Cython.~~", follow the below command.
    ```Shell
    sudo pip install cython
    ```
    
### Update
The updated code will be uploaded on [git server](https://github.com/yochin/smallcorgi_Faster-RCNN_TF_yochin.git).

###References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)

