# DNN Facial Landmark Thermometer
## GPU-Accelerated Facial Landmark Thermometer for Jetson Nano & PureThermal Mini using FLIR Radiometric Lepton 3.5

![startupExample](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/startupExample.gif)

### 1. To begin, update your Jetson and install dependencies by running the following:
```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 install dlib
pip3 install Pillow
```


### 2. This project requires OpenCV to have CUDA compiled for GPU-acceleration, run the following:
```
git clone https://github.com/mdegans/nano_build_opencv.git
cd nano_build_opencv/
./build_opencv.sh
```


### 3. Finally to run:

```
git clone https://github.com/Axeiru/DNNFacialLandmarkThermometer.git
cd DNNFacialLandmarkThermometer/
python3 DNNFacialLandmarkThermometer.py
```

![example_3](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/example_3.gif)

### Copies of required files are included, however original sources are listed below and deserve many thanks!
deploy.prototxt.txt: [source.](https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt) - MIT License

res10_300x300_ssd_iter_140000.caffemodel: [source.](https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel) - MIT License

shape_predictor_68_face_landmarks.dat: [source.](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) - Creative Commons Zero v1.0



### More Examples:
![example_1](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/example_1.gif)

![example_2](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/example_2.gif)


