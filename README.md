# yolov7-pose-estimation
This model will save the keypoint detections in JSON File 


### Steps to run Code
```
- Clone the repository.
```
git clone https://github.com/Avishaek/Yolov7_Pose-Estimation.git
```

- Goto the cloned folder.
```
cd yolov7-pose-estimation
```
```
pip install -r requirements.txt
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}
 OR 

curl -L https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt -o yolov7-w6-pose.pt
```

- Run the code with mentioned command below.
```
python pose-estimate.py

#if you want to change source file
python pose-estimate.py --source "your custom video.mp4"

#For CPU
python pose-estimate.py --source "your custom video.mp4" --device cpu

#For GPU
python pose-estimate.py --source "your custom video.mp4" --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python pose-estimate.py --source "your IP Camera Stream URL" --device 0

#For WebCam
python pose-estimate.py --source 0

#For External Camera
python pose-estimate.py --source 1
```
EXAMPLE
```
python pose-estimate.py --source "/content/videoplayback.mp4" --device 0
```
- Output file will be created in the working directory with name <b>["your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### RESULTS


<img src="https://github.com/Avishaek/Yolov7_Pose-Estimation/blob/main/Detection.png">

