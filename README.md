# DNN Facial Landmark Thermometer
## GPU-Accelerated Facial Landmark Thermometer for Jetson Nano & PureThermal Mini using FLIR Radiometric Lepton 3.5
### This project attempts to detect fevers by tracking the medial canthus of the eye which serve as a proxy for internal body temperature.

<br/>

![mainExample_4](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Main%20Examples/mainExample_4.gif)

![mainExample_1](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Main%20Examples/mainExample_1.gif)

<br/>

### Installation:

#### 1. To begin, update your Jetson and install dependencies by running the following:
```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
pip3 install dlib
pip3 install Pillow
```
#### 2. This project requires OpenCV to have CUDA compiled for GPU-acceleration, run the following:
```
git clone https://github.com/mdegans/nano_build_opencv.git
cd nano_build_opencv/
./build_opencv.sh
```
#### 3. Finally to run:

```
git clone https://github.com/Axeiru/DNNFacialLandmarkThermometer.git
cd DNNFacialLandmarkThermometer/
python3 DNNFacialLandmarkThermometer.py
```

<br/>

![mainExample_2](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Main%20Examples/mainExample_2.gif)

<br/>

### Working Principles:

Visible and thermal images contain radically different information that often do not correlate. This issue is side-stepped by performing Canny edge-detection and finding the (x, y) offsets between images which maximize 'similiarity' between detected edges. The images are aligned according to the coordinates of the maximum computed cross-correlation. 

<br/>

![mainExample_3](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Main%20Examples/mainExample_3.gif)

<br/>

### Usage:

- This program displays 9 views:
```
[[visibleImageDetectedFaces, overlayedView, thermal_image],
[visibleCanny, cannyOverlayedView, thermalCanny],
[crossCorrelationFrame, croppedOverlayedAligned, simulatedCroppedOverlayedCanny]]
 ```
- The medial canthus temperature of each detected face can be labelled in real time by toggling "Display Medial Canthus Temperatures"

- The displayed views can be saved as a video or as individal frames.

- Min/max temperatures are labelled on ```thermal_image```

- Spot temperature readings are labelled on ```visibleImageDetectedFaces``` and ```visibleImageDetectedFaces``` at the mouse cursor's position

- The opacity of ```overlayedView``` and ```croppedOverlayedAligned``` can be controlled via the opacity slider.

- Canny edge detection hysteresis thresholds can be controlled for each image via sliders.

- The background is norminally white, however, it switches between:
  - Green, if faces are detected and all medial canthus temperatures are below the fever threshold of 38 degrees Celcius
  - Red, if a fever is detected, i.e. medial canthus temperature >= 38.0 Celcius.

<br/>

![stillTrackingExample_2](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Main%20Examples/stillTrackingExample_2.jpg)

<br/>

### Future Avenues for Image Registration Improvement:

This implementation is entirely linear and naive. It does template-matching which is a simple sweep between images, it cannot account for the nonlinear distortion present in optical lenses. Furthermore, it only applies image offsets and is less expressive than even an affine transformation as no rotation or scaling is attempted. Ideally a non-linear projection between captured images would be used.

Currently, this implementation naively chooses the coordinates of the maximum similarity, when better heuristics may exist:

- A momentum-based model could be used to prevent discontinuitues in image offsets, ensuring smooth adjustments

- Another possibility is applying a Gaussian mask to the computed cross-correlation to prefer smaller, more central offsets.

<br/>

### Framerate Performance Notes:

- Ideally, images would be captured from each camera in separate threads, however, tkinter is inherently single-threaded. Future revisions may move away from tkinter to implement proper multithreaded image capture.

- The FLIR Lepton 3.5's maximum capture rate is ~9Hz which presents an artifical 'sweet-spot,' though higher visible framerates still appear smoother. A Jetson Xavier NX is likely better suited at running the current implementation at useable framerates than a Jetson Nano.

- Presently the most computationally intensive operations performed are Canny edge-detection and the cross-correlation template matching. The most direct way to improve performance is by reducing ```self.alignment_padding``` to smaller values. This will have the effect of reducing the maximum possible offset.

<br/>  

### General Notes:

- Overclocking is recommended to attain maximum performance, however, this requires recompiling the kernel and takes several hours. A procedure to overclock the Jetson Nano's CPU to 2GHz & GPU to 1.2GHz will be added.

- This project was entirely developed on a Jetson Nano, including recording & transcoding examples.

- This project is built upon the stock Jetson Nano 4GB Developer Kit SD Card Image (JetPack 4.6) released by NVIDIA. Apart from building OpenCV from source for CUDA support, all dependencies are included in the stock image.

- Future offloading of processing to the GPU should generally improve performance, however, Jetson Xavier NX may be a better choice for future revisions as the Nano's CPU is used heavily and showing its limits.

- A Jetson Nano 4GB is highly recommended as the facial detection model and matrix operations consume ~1GB of RAM. This project may run on the Jetson Nano 2GB, but this hasn't been tested yet and may result in severe memory thrashing.

<br/>

### Planned improvements/features:

- Manually setting high-gain mode & skin emissivity values via firmware
- Implementing command line options
- Custom colormaps, resolutions, views
- Logging to file(# of detected faces, temperatures, general statistics, etc.)

<br/>

![example_3](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Other%20Examples/example_3.gif)

<br/>

### Supporting Resources:

*Multispectral Thermal Imaging with a LWIR+VIS Camera*

- Hines, Jacob, and Evan Wang. “Multispectral Thermal Imaging with a LWIR+VIS Camera.” *Stanford EE367 / CS448I: Computational Imaging*, Stanford University, 17 Mar. 2019, http://stanford.edu/class/ee367/Winter2019/hines_wang_report.pdf.

  - Extremely illuminating and informative survey into multispectral image registration and fusion. Many thanks to the Authors!
  

<br/>

The included stl can be printed to serve as an alignment mount for the dual cameras:

<img src="https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/dual_cam_module_v7.png" width="500" />

<br/>

- Copies of required files are included, however, original sources are listed below and deserve many thanks!
  - deploy.prototxt.txt: [source.](https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt) - MIT License

  - res10_300x300_ssd_iter_140000.caffemodel: [source.](https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel) - MIT License

  - shape_predictor_68_face_landmarks.dat: [source.](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) - Creative Commons Zero v1.0


<br/>

### More Examples:
#### Samples are also provided in original resolution
![example_1](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Other%20Examples/example_1.gif)

![example_2](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Other%20Examples/example_2.gif)

![startupExample](https://github.com/Axeiru/DNNFacialLandmarkThermometer/blob/main/examples/Other%20Examples/startupExample.gif)