import cv2
import dlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageTk
import time
import timeit
import tkinter

def loadModel():
    # Loading GPU-accelerated model
    print("[INFO] Loading DNN model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def initializeCustomColormap():
    # Builds color-on-gray custom colormap
    print("[INFO] Initializing Colormap...")
    colorMap1 = plt.cm.binary(np.linspace(0, 1, 128))
    colorMap2 = plt.cm.inferno(np.linspace(0, 1, 128))
    colors = np.vstack((colorMap1, colorMap2))
    customColorMap = mcolors.LinearSegmentedColormap.from_list("CustomColorMap", colors)
    return customColorMap


def ctof(val):
    # Celsius to fahrenheit
    return 1.8 * ktoc(val) + 32.0


def ktoc(val):
    # Kelvin to celcius
    return (val - 27315) / 100.0


def raw_to_8bit(data):
    # Preprocesses raw thermal pixel data: normalizes and bitshifts into a single byte to be represented as a black and white image
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


def displayTemperature(img, tempKelvin, position, color):
    # Labels temperature onto image
    val = ctof(tempKelvin)
    output = cv2.putText(img, "{0:.1f} degF".format(val), position, cv2.FONT_HERSHEY_COMPLEX, 0.75, color, 2)
    x, y = position
    output = cv2.line(output, (x - 2, y), (x + 2, y), color, 2)
    output = cv2.line(output, (x, y - 2), (x, y + 2), color, 2)
    return output


def displayFaceTemperature(img, tempKelvin, position, color, size, displayMedialCanthusTemp):
    # Label temperature onto image, specifc to faces
    val = ctof(tempKelvin)
    if displayMedialCanthusTemp.get():
        output = cv2.putText(img, str('{:g}'.format(float('{:.{p}g}'.format(val, p=3)))), (position[0]-20, position[1]-20), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2) ##'p=3' represents number of sigfigs to display
    x, y = position
    output = cv2.line(img, (x - 2, y), (x + 2, y), color, size)
    output = cv2.line(output, (x, y - 2), (x, y + 2), color, size)
    return output


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=8.7,
    flip_method=6,):  
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true sync=false"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def detectFaces(self, visible_image, thermal_image, thermal_data):
    # Main logic loop, detects faces, labels them on the output image and looks up temperature from thermal camera
    # returns labelled images, and true if fever detected
    dnnfaces = []
    (h, w) = visible_image.shape[:2]
    feverDetected = False
    blob = cv2.dnn.blobFromImage(cv2.resize(visible_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    self.net.setInput(blob)
    detections = self.net.forward()

    # Loops over detected faces, draws box and labels with confidence score
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < .5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        offset = 5
        
        startX-=offset
        startY-=offset
        endX+=offset
        endY+=offset

        (xtopLeft, ytopLeft, xbottomRight, ybottomRight) = (startX, startY, endX, endY)
        dlibRect = dlib.rectangle(xtopLeft, ytopLeft, xbottomRight, ybottomRight)

        dnnfaces.append(dlibRect)

        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(visible_image, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(visible_image, text, (startX, y), cv2.FONT_HERSHEY_COMPLEX , 1, (0, 255, 0), 2)

    # Draws 68 facial landmarks on visible image, labels thermal image with medial canthus points as well as measured temperature
    for face in dnnfaces:
        face_landmarks = self.dlib_facelandmark(visible_image, face)
        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
        
            y_bound, x_bound = thermal_image.shape[:2]
            if 0 < x < x_bound and 0 < y < y_bound:
                try:
                    cv2.circle(visible_image, (x, y), 1, (255, 255, 255), 1)
                    if n == 39 or n == 42:
                        # Left and right inner corners of eyes
                        raw_value = thermal_data[y, x]
                        visible_image = displayFaceTemperature(visible_image, raw_value, (x,y), (0, 0, 255), 1, self.displayMedialCanthusTemp)
                        thermal_image = displayFaceTemperature(thermal_image, raw_value, (x,y), (0, 0, 0), 3, self.displayMedialCanthusTemp)
                        celciusTemp = ktoc(raw_value)
                        if celciusTemp >= 38.0:
                            self.window.configure(bg='red')
                            feverDetected = True
                        else:
                            self.window.configure(bg='forest green')
                            
                except Exception as e:
                    print(e)
                    print("something's wrong!!")
                    print("x, y: ", x, y)
                    print("x_bound, y_bound: ", x_bound, y_bound)
    if not dnnfaces:
        self.window.configure(bg='white')

    return (visible_image, thermal_image, feverDetected)


class GUI:
    def __init__(
        self,
        window,
        windowTitle,
        visibleCam = (gstreamer_pipeline(flip_method=6), cv2.CAP_GSTREAMER, False),
        thermalCam = ("/dev/video1", cv2.CAP_V4L2, True),
    ):
        # Main program variables/options & initialization values
        self.net = loadModel()
        self.frameWidth = 640
        self.frameHeight = 360
        self.alignment_padding = 50
        self.window = window
        self.window.title(windowTitle)
        self.dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.currentFrame = None
        self.outputPhoto = None
        self.mousePosition = None
        self.recording = False
        self.outputRecording = None
        self.outputText = "FPS: "
        self.window.bind("<equal>", lambda x: self.opacitySlider.set(self.opacitySlider.get() + 1) if (self.opacitySlider.get() <= 99) else (self.opacitySlider.set(100))),
        self.window.bind("<minus>", lambda x: self.opacitySlider.set(self.opacitySlider.get() - 1) if (self.opacitySlider.get() >= 1) else (self.opacitySlider.set(0))),
        
        self.finalOutputFrameResizeRatio = 1.5

        self.thermalCamSource = thermalCam
        self.visibleCamSource = visibleCam

        self.vid_visible = MyVideoCapture(self.visibleCamSource)
        self.vid_thermal = MyVideoCapture(self.thermalCamSource)

        self.canvas = tkinter.Canvas(window, width=1280, height=720)
        self.canvas.pack(pady = 5)

        self.stringFPS = tkinter.StringVar()
        self.label = tkinter.Label(window, textvariable=self.stringFPS)
        self.stringFPS.set(self.outputText)
        self.label.pack()

        self.displayMedialCanthusTemp = tkinter.BooleanVar(value=False)
        self.displayMedialCanthusTempCheckBox = tkinter.Checkbutton(window, text = "Display Medial Canthus Temperatures", variable=self.displayMedialCanthusTemp)
        self.displayMedialCanthusTempCheckBox.pack()

        self.snapshotButton = tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.snapshotButton.pack(anchor=tkinter.CENTER, pady = 5)

        self.recordVideoButton = tkinter.Button(window, text="Record", width=50, command=self.recordVideo)
        self.recordVideoButton.pack(anchor=tkinter.CENTER, pady = 10)

        self.opacitySlider = tkinter.Scale(window, length = 300, from_= 0, to = 100, orient=tkinter.HORIZONTAL, label = "Overlay Opacity: ")
        self.opacitySlider.set(50)
        self.opacitySlider.pack(pady = 5)

        self.sliderFrame = tkinter.Frame(window)
        self.sliderFrame.pack()

        self.visibleMinSlider = tkinter.Scale(self.sliderFrame, length = 300, from_= 0, to = 400, orient=tkinter.HORIZONTAL, label = "visibleMinSlider: ")
        self.visibleMinSlider.set(200)
        self.visibleMinSlider.pack(padx = 5, pady = 5, side = tkinter.LEFT)

        self.visibleMaxSlider = tkinter.Scale(self.sliderFrame, length = 300, from_= 0, to = 400, orient=tkinter.HORIZONTAL, label = "visibleMaxSlider: ")
        self.visibleMaxSlider.set(235)
        self.visibleMaxSlider.pack(padx = 5, pady = 5, side = tkinter.LEFT)

        self.thermalMinSlider = tkinter.Scale(self.sliderFrame, length = 300, from_= 0, to = 400, orient=tkinter.HORIZONTAL, label = "thermalMinSlider: ")
        self.thermalMinSlider.set(0)
        self.thermalMinSlider.pack(padx = 5, pady = 5, side = tkinter.RIGHT)

        self.thermalMaxSlider = tkinter.Scale(self.sliderFrame, length = 300, from_= 0, to = 400, orient=tkinter.HORIZONTAL, label = "thermalMaxSlider: ")
        self.thermalMaxSlider.set(275)
        self.thermalMaxSlider.pack(padx = 5, pady = 5, side = tkinter.RIGHT)

        self.finalImage = self.canvas.create_image(0, 0, image=None, anchor=tkinter.NW)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Writes last successful frame to SD, Note: ./output/ must exist
        cv2.imwrite("output/frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg",cv2.cvtColor(self.currentFrame, cv2.COLOR_RGB2BGR))

    def recordVideo(self):
        # Displays recording status, has noticeable impact on processed FPS, Note: ./output/ must exist
        if self.recording:
            self.recording = False
            self.recordVideoButton.configure(text="Record", bg="white", activebackground="white")
            self.outputRecording.release()
        else:
            self.recording = True
            self.recordVideoButton.configure(text="Recording", bg="red", activebackground="red")
            self.outputRecording = cv2.VideoWriter("output/vid-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".avi",
            cv2.VideoWriter.fourcc('M','J','P','G'),
                3, # Empirically determined based on Jetson Nano 4GB, CPU @ 2GHz, GPU @ 1.2GHz, recording to main SD card, stock performance will be lower
                (1280, 720))

    def readMousePosition(self):
        # Calculates and returns mouse position based on predefined image dimensions and scales
        pointerX, pointerY = self.canvas.winfo_pointerxy()
        windowX = self.canvas.winfo_rootx()
        windowY = self.canvas.winfo_rooty()

        relativeX = int(((pointerX - windowX) % int(self.frameWidth/self.finalOutputFrameResizeRatio))*self.finalOutputFrameResizeRatio)
        relativeY = int((pointerY - windowY)*self.finalOutputFrameResizeRatio)
        if relativeY > (self.frameHeight - 1):
            relativeY = int(self.frameHeight - 1)

        return (relativeX, relativeY)

    def templateBasedImageRegistration(self, visibleCanny, thermalCanny, visible_image, thermal_image):
        # Performs template matching between detected canny edges in visible and thermal camera images then naively aligns based on coordinates of max computed cross-correlation
        # Returns third row frame by concatenating: crossCorrelationFrame, croppedOverlayedAligned, croppedOverlayedCannyFrame horizontally
        paddedVisibleCanny = np.pad(visibleCanny, pad_width=[(self.alignment_padding, self.alignment_padding), (self.alignment_padding, self.alignment_padding),(0, 0)], mode='constant')
        templateMatchResult = cv2.matchTemplate(paddedVisibleCanny, thermalCanny, cv2.TM_CCORR)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(templateMatchResult)
        top_left = max_loc
        bottom_right = (top_left[0] + self.frameWidth, top_left[1] + self.frameHeight)

        oneSidedPaddedThermal = np.pad(thermal_image, pad_width=[(top_left[1], 0), (top_left[0], 0), (0,0)], mode='constant')
        oneSidedPaddedVisible = np.pad(visible_image, pad_width=[(self.alignment_padding, self.alignment_padding), (self.alignment_padding, self.alignment_padding), (0,0)], mode='constant')

        delta_w = oneSidedPaddedVisible.shape[0] - oneSidedPaddedThermal.shape[0]
        delta_h = oneSidedPaddedVisible.shape[1] - oneSidedPaddedThermal.shape[1]

        oneSidedPaddedThermal = np.pad(oneSidedPaddedThermal, pad_width=[(0, delta_w), (0, delta_h), (0,0)], mode='constant')
        overlayedAligned = cv2.addWeighted(oneSidedPaddedThermal, (self.opacitySlider.get()/100), oneSidedPaddedVisible, 1-(self.opacitySlider.get()/100), 0)
        croppedOverlayedAligned = overlayedAligned[self.alignment_padding:-self.alignment_padding, self.alignment_padding:-self.alignment_padding, :]

        calculatedRegistration = cv2.rectangle(paddedVisibleCanny,top_left, bottom_right, 255, 1)
        croppedOverlayedCannyFrame = calculatedRegistration[self.alignment_padding:-self.alignment_padding,self.alignment_padding:-self.alignment_padding,:]
        croppedOverlayedCannyFrame = cv2.addWeighted(croppedOverlayedCannyFrame, 1, thermalCanny, 1, 0)

        templateMatchResult = cv2.merge((templateMatchResult, templateMatchResult, templateMatchResult))
        templateMatchResult = cv2.circle(templateMatchResult, max_loc, radius=5, color=(255, 0, 0), thickness=-1)

        crossCorrelationFrame = np.full((self.frameHeight, self.frameWidth, 3), 255, dtype=np.uint8)
        crossCorrelationFrame[(self.frameHeight//2 - self.alignment_padding):(2*(self.alignment_padding)+self.frameHeight//2+1 - self.alignment_padding), (self.frameWidth//2 - self.alignment_padding):(2*(self.alignment_padding)+self.frameWidth//2+1 - self.alignment_padding), :] = templateMatchResult

        finalAlignmentFrame = np.concatenate((crossCorrelationFrame, croppedOverlayedAligned, croppedOverlayedCannyFrame), axis=1)

        return finalAlignmentFrame

    def cannyBasedImageRegistration(self, visible_image, thermal_image):
        # Performs canny edge detection on the visible and thermal frames, the hysteresis thresholding is based on max/min slider user input 
        # returns second row frame by concatenating: visibleCanny, cannyOverlayedView, thermalCanny horizontally
        visibleCanny = cv2.Canny(visible_image, self.visibleMinSlider.get(), self.visibleMaxSlider.get(), L2gradient=True)
        thermalCanny = cv2.Canny(thermal_image, self.thermalMinSlider.get(), self.thermalMaxSlider.get(), L2gradient=True)

        whiteVisibleCanny = cv2.merge((visibleCanny, visibleCanny, visibleCanny))
        whiteThermalCanny = cv2.merge((thermalCanny, thermalCanny, thermalCanny))

        blankImage = np.zeros((visibleCanny.shape[0], visibleCanny.shape[1]), np.uint8)
        visibleCanny = cv2.merge((visibleCanny, blankImage, visibleCanny))
        thermalCanny = cv2.merge((blankImage, thermalCanny, blankImage))

        cannyOverlayedView = cv2.addWeighted(visibleCanny, 1, thermalCanny, 1, 0)

        finalCannyFrame = np.concatenate((visibleCanny, cannyOverlayedView, thermalCanny), axis=1)
        finalAlignmentFrame = self.templateBasedImageRegistration(whiteVisibleCanny, whiteThermalCanny, visible_image, thermal_image)

        return finalCannyFrame, finalAlignmentFrame

    def update(self):
        # Main Tkinter window update logic as a single thread, optimally would capture camera frames and have GUI in separate threads, but would only help up to 9 FPS due to Lepton limitation
        # Framerate is calculated by time it takes to complete an iteration of update()
        start = timeit.default_timer()
        thermal_ret, thermal_image, thermal_data = self.vid_thermal.get_frame()
        visible_ret, visible_image = self.vid_visible.get_frame()

        if thermal_ret and visible_ret:
            self.mousePosition = self.readMousePosition()
            mouseTemp = thermal_data[self.mousePosition[1], self.mousePosition[0]]

            finalCannyFrame, finalAlignmentFrame = self.cannyBasedImageRegistration(visible_image, thermal_image)

            visible_image = displayTemperature(visible_image, mouseTemp, self.mousePosition, (34, 250, 154))
            thermal_image = displayTemperature(thermal_image, mouseTemp, self.mousePosition, (34, 139, 34))
            
            visibleImageDetectedFaces, thermal_image, detectedFever = detectFaces(self, visible_image, thermal_image, thermal_data)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)
            thermal_image = displayTemperature(thermal_image, minVal, minLoc, (0, 0, 255))
            thermal_image = displayTemperature(thermal_image, maxVal, maxLoc, (255, 0, 0))
            
            overlayedView = cv2.addWeighted(thermal_image, (self.opacitySlider.get()/100), visibleImageDetectedFaces, 1-(self.opacitySlider.get()/100), 0)
            finalOverlayedViewFrame = np.concatenate((visibleImageDetectedFaces, overlayedView, thermal_image), axis=1)
            finalOutputFrame = np.concatenate((finalOverlayedViewFrame, finalCannyFrame), axis=0)
            finalOutputFrame = np.concatenate((finalOutputFrame, finalAlignmentFrame), axis=0)

            finalOutputFrame = cv2.resize(finalOutputFrame, dsize=(int(finalOutputFrame.shape[1]/self.finalOutputFrameResizeRatio), int(finalOutputFrame.shape[0]/self.finalOutputFrameResizeRatio)), interpolation=cv2.INTER_CUBIC)
            
            self.currentFrame = finalOutputFrame
            self.outputPhoto = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(finalOutputFrame))
            if self.recording:
                self.outputRecording.write(cv2.cvtColor(finalOutputFrame,cv2.COLOR_RGB2BGR))

            self.canvas.itemconfigure(self.finalImage, image = self.outputPhoto)
        
        stop = timeit.default_timer()
        strFPS = str(round(1/(stop - start), 2))
        self.outputText = ('FPS: ' + strFPS)

        if detectedFever:
            self.outputText = (self.outputText + " - Fever Detected!")
            
        self.stringFPS.set(self.outputText)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.camera_settings = video_source
        self.vid = cv2.VideoCapture(video_source[0], video_source[1])
        if video_source[2]: # If thermal
            self.customColorMap = initializeCustomColormap()
            print("[INFO] Thermal cam setting set", video_source[2])
            if self.vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("Y", "1", "6", " ")):
                print("[INFO] Success Setting CAP_PROP_FOURCC")
            if self.vid.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                print("[INFO] Success Setting CAP_PROP_CONVERT_RGB")
            if self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 1):
                print("[INFO] Success Setting CAP_PROP_BUFFERSIZE")

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                if self.camera_settings[2]:
                    colormap = self.customColorMap
                    thermal_data = cv2.flip(cv2.resize(frame[:, :], (640, 480))[59:419, :], 1)
                    singleChannelThermalImage = cv2.cvtColor(raw_to_8bit(thermal_data.copy()), cv2.COLOR_BGR2GRAY)
                    thermal_image = (colormap(singleChannelThermalImage) * 2**8).astype(np.uint8)[:,:,:3]
                    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_RGBA2RGB)
                    return (ret, thermal_image, thermal_data)
                else:
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (False, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
guiWindow = tkinter.Tk()
guiWindow.geometry("1920x1080")
guiWindow.bind("<Escape>", lambda x: guiWindow.destroy())

GUI(guiWindow, "DNN Facial Landmark Thermometer: PureThermal Mini - FLIR Radiometric Lepton 3.5")                                    