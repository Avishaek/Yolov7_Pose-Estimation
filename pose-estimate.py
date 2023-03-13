import cv2
import time
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer

@torch.no_grad()
def run(
        poseweights='yolov7-w6-pose.pt',
        source='football1.mp4',
        device='cpu'):
    
    #list to store time
    time_list = []
    #list to store fps
    fps_list = []
    
    #select device
    device = select_device(opt.device)
    half = device.type != 'cpu'
    
    # Load model
    model = attempt_load(poseweights, map_location=device)  # load FP32 model
    _ = model.eval()

    #video path
    video_path = source

    #pass video to videocapture object
    cap = cv2.VideoCapture(video_path)

    #check if videocapture not opened
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    #get video frame width
    frame_width = int(cap.get(3))

    #get video frame height
    frame_height = int(cap.get(4))

    #code to write a video
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{out_video_name}_keypoint.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (resize_width, resize_height))

    #count no of frames
    frame_count = 0
    #count total fps
    total_fps = 0 
    arr_list=[]
    json_resultss= []
    #loop until cap opened or video not complete
    while(cap.isOpened):
        
        print("Frame {} Processing".format(frame_count))
        
        #get frame and success from video capture
        ret, frame = cap.read()
        #if success is true, means frame exist
        if ret:
            
            #store frame
            orig_image = frame

            #convert frame to RGB
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            
            #convert image data to device
            image = image.to(device)
            
            #convert image to float precision (cpu)
            image = image.float()
            
            #start time for fps calculation
            start_time = time.time()
            
            #get predictions
            with torch.no_grad():
                output, _ = model(image)

            #Apply non max suppression
            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)

            
            im0 = image[0].permute(1, 2, 0) * 255
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            #reshape image format to (BGR)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            for idx in range(output.shape[0]):
                plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
                
                #Plotting key points on Image
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.rectangle(im0,(int(xmin), int(ymin)),(int(xmax), int(ymax)),color=(255, 0, 0),
                    thickness=1,lineType=cv2.LINE_AA)
                #cv2.putText(im0, f'{"ID".upper()} {output[idx][1]}', (xmin, ymax), font, 1.4, [255, 0, 255], 2)
                if len(idx) >1:
                    cv2.putText(im0, f'ID: {int(idx)}', (xmin, ymax), font, 1.4, [255, 0, 255], 2) 
            
            #Calculatio for FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            #append FPS in list
            fps_list.append(total_fps)
            
            #append time in list
            time_list.append(end_time - start_time)
            
            #add FPS on top of video
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            
            # cv2.imshow('image', im0)
            out.write(im0)

            # Convert every keypoint detection in frame to List
            skeletons= output
            for idx in range(output.shape[0]):
              skeletons[idx][0]= frame_count
              skeletons[idx][1]= idx+1
              skeletons= skeletons
              json_result = {
                "frame": frame_count,
                "keypoints": skeletons[idx,7:58].tolist(),
                "score": skeletons[idx,6].tolist(),
                "box": skeletons[idx, 2:6].tolist(),
                "idx": ["To be calculated using Deepsort"]
              }
              #json_resultss=json_resultss.tolist()
              json_resultss.append(json_result) 
            kpts= skeletons.tolist()
            arr_list.extend(kpts)        
            

            
             
            print("No. of skeletons:", output.shape[0])
            print("Frame count is:", frame_count)
            print(output)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    # cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
   
    #plot the comparision graph
    plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)

    # Write the list to a JSON file
    with open('keypoints.json', 'w') as f:
        json.dump(arr_list, f)

    with open('keys.json', 'w') as f:
        json.dump(json_resultss, f)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
