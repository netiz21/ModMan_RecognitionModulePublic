# ModMan object detection and pose estimation module

This is an object detection and pose estimation module in Modular Manipulation Project.

The object detection part is mainly based on the [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF) that refers to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.
Pose estimation part is composed of SURF-based feature matching and PnPSolver using [OpenCV](https://opencv.org/)

Additional parts such as various input modules (web camera, Intel realsense, images and videos) and communication modules are added to the main code. 

### Requirements: hardware

For running the recognition module, GPU memory is needed.

### Installation: software

1. Install Ubuntu 16.04.xx
 
2. Install NVIDIA driver, CUDA 8.0, cudnn 5.1.

    2.1. Nvidia driver

    ```Shell
    sudo apt-get purge nvidia*
    sudo add-apt-repository ppa:graphics-drivers
    sudo apt-get update

    sudo apt-get install nvidia-375

    lsmod | grep nvidia
    ```

    2.2. CUDA 8.0
    
    Access https://developer.nvidia.com/cuda-downloads
    
    Download Legacy Releases->CUDA Toolkit 8.0 GA1 (Sept 2016)->Linux->x86_64->Ubuntu0>16.04->deb(local)->download

    ```Shell
    sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda
    
    ls /usr/local/cuda/lib64
    ```
    If you find the file names "~~~.so.8.0", the installation is completed.
    
    2.3. CUDNN 5.1
    
    Access https://developer.nvidia.com/cudnn
    
    Download Download -> Log in -> Survey (pass) -> Download cuDNN v5.1 for CUDA 8.0 -> cuDNN v5.1 Library for Linux
    or Download from [here](https://drive.google.com/open?id=1o7sZdUlJp6H8ZXhBN3IrukM0HbqrCnPj).
    
    ```Shell
    tar -zxvf cudnn-8.0-linux-x64-v6.0.tgz
    sudo cp ./cuda/include/* /usr/local/cuda-8.0/include/
    sudo cp ./cuda/lib64/* /usr/local/cuda-8.0/lib64/

    ls /usr/local/cuda/lib64/libcudnn*

    ```
    
    If you find the file names "~~~.so.5.1.x", the installation is completed.

3. Anaconda installation and import conda environment

    Download Anaconda at https://www.anaconda.com/download/#linux with Python 2.7 version.
    Install Anaconda.
    
    Download [the conda env file](https://drive.google.com/file/d/1xfBrtvyViyP9UWn7mS1mnJ_4EFxh7BiM/view?usp=sharing).
    Import ModMan conda env.

    ```Shell
    conda env create -f ModMan_env.yml
    ```

4. Download and compile Faster-RCNN_TF based ModMan recognition module

    ```Shell    
    git clone https://github.com/yochin/smallcorgi_Faster-RCNN_TF_yochin.git
    ```
    ```Shell    
    source activate ModMan
    cd $FRCN/lib        // Let $FRCN as the downloaded folder name.
    make
    ```
    
    If you meet an error while compiling, then see the below solutions.
        
    * error: roi_pooling_op.cu.o: No such file or directory
    
      First, check the version of CUDA and cudnn. Then, add to your path(or you can also add the path in ~/.bashrc.).
    
      ```Shell
      export PATH="/usr/local/cuda-8.0/bin:$PATH"
      ```
    
    * error: fatal error: math_functions.hpp: No such file or directory
    
      [Ref: https://github.com/tensorflow/tensorflow/issues/15389]
    
      Make a softlink from /usr/local/cuda-9.1/include/crt/math_functions.hpp to /usr/local/cuda-9.1/include/math_functions.hpp
    
      ```Shell
      cd /usr/local/cuda-9.1/include
      ln -s ./crt/math_functions.hpp ./math_functions.hpp
      ```
      
    * error: #error Do not use this file, it is the result of a failed Cython compilation.
    
      [Ref: https://github.com/rbgirshick/py-faster-rcnn/issues/647]
    
      Install cython, then make again.
    
      ```Shell
      source activate ModMan
      conda install -c anaconda cython
      ```
    
5. Download trained model

    Download [DBv1](https://drive.google.com/open?id=1whjx999HjnITSwtCuP849gHVsOz8Ly2S) and [models](https://drive.google.com/open?id=1tVcE0uufb4D5XnUO34HWoqJr2pBainy9).

    Move DBv1 folder to $FRCN/yochin_tools/PoseEst
    
    Move models folder to $FRCN

6. Set paths and run the program

    In $FRCN/yochin_tools/yo_network_info.py, change the variable PATH_BASE to become the real path to $FRCN.
    
    In $FRCN/yochin_tools/my_demo_tf_wPoseEst_conSKKUKIST.py (this is the main code), edit followings.

    If you don't want to use realsense, then comment line 41.
    
    ```Shell
    # import pyrealsense as pyrs
    ```
    
    * If you use the program as a server (receiving images from the client PC), then    
    
      In line 449, INPUT_TYPE = 5
      
      In line 459, extMat = getCamIntParams('client')
      
      In line 458 and 459, AR_IP(IP address) and AR_PORT(port number) should be set.
    
    * If you use the program as a standalone program using web camera,
    
      In line 449, INPUT_TYPE = 0
      
      In line 459, extMat = getCamIntParams('') <-- select the proper camera name.

    * If you set all parameters, then run the program.
      ```Shell
      python my_demo_tf_wPoseEst_conSKKUKIST.py
      ```
    
### Errors

If you meet a error message like 'import module error ###', then install that module using one of below commands.

```Shell
source activate ModMan

pip install pyyaml
conda install -c auto easydict
conda install -c auto scipy
conda install -c https://conda.anaconda.org/menpo opencv
conda install -c conda-forge hdf5 
conda install -c auto matplotlib
conda install -c anaconda h5py
```

### Update
The updated code will be uploaded on [git server](https://github.com/yochin/smallcorgi_Faster-RCNN_TF_yochin.git).

###References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)

