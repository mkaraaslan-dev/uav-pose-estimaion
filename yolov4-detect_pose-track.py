import sys
import cv2
import numpy as np
import argparse
import random
import time

from dlclive import DLCLive, Processor

dlc_proc = Processor()
dlc_live = DLCLive("exported-models/DLC_uav_pose_resnet_50_iteration-0_shuffle-1", processor=dlc_proc)


x_size = 229
y_size = 174
img_back = np.zeros([y_size,x_size,3],dtype=np.uint8)
dlc_live.init_inference(img_back)

class YOLOv4:

    def __init__(self):
        """ Method called when object of this class is created. """

        self.args = None
        self.net = None
        self.names = None

        self.parse_arguments()
        self.initialize_network()
        self.run_inference()

    def parse_arguments(self):
        """ Method to parse arguments using argparser. """

        parser = argparse.ArgumentParser(description='Object Detection using YOLOv4 and OpenCV4')
        parser.add_argument('--image', type=str, default='', help='Path to use images')
        parser.add_argument('--stream', type=str, default='a.mp4', help='Path to use video stream')
        parser.add_argument('--cfg', type=str, default='models/yolov4-tiny-obj.cfg', help='Path to cfg to use')
        parser.add_argument('--weights', type=str, default='models/yolov4-tiny-obj_best.weights', help='Path to weights to use')
        parser.add_argument('--namesfile', type=str, default='models/obj.names', help='Path to names to use')
        parser.add_argument('--input_size', type=int, default=224, help='Input size')
        parser.add_argument('--use_gpu', default=False, action='store_true', help='To use NVIDIA GPU or not')

        self.args = parser.parse_args()

    def initialize_network(self):
        """ Method to initialize and load the model. """

        self.net = cv2.dnn_DetectionModel(self.args.cfg, self.args.weights)
        
        if self.args.use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        if not self.args.input_size % 32 == 0:
            print('[Error] Invalid input size! Make sure it is a multiple of 32. Exiting..')
            sys.exit(0)
        self.net.setInputSize(self.args.input_size, self.args.input_size)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(self.args.namesfile, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

    def image_inf(self):
        """ Method to run inference on image. """

        frame = cv2.imread(self.args.image)

        timer = time.time()
        classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        print('[Info] Time Taken: {}'.format(time.time() - timer), end='\r')
        
        if(not len(classes) == 0):
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%s: %.2f' % (self.names[classId], confidence)
                left, top, width, height = box
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                cv2.rectangle(frame, box, color=(b, g, r), thickness=2)
                cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
                cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)

        cv2.imwrite('result.jpg', frame)
        cv2.imshow('Inference', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return

    def stream_inf(self):
        """ Method to run inference on a stream. """

        source = cv2.VideoCapture(0 if self.args.stream == 'webcam' else self.args.stream)

        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)

        while(source.isOpened()):
            ret, frame = source.read()
            if ret:
                timer = time.time()
                classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
                try:
                    print('[Info] Time Taken: {} | FPS: {}'.format(time.time() - timer, 1/(time.time() - timer)), end='\r')
                except:
                    pass
                if(not len(classes) == 0):
                    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                        label = '%s: %.2f' % (self.names[classId], confidence)
                        left, top, width, height = box
                        b = 0
                        g = 0
                        r = 255
                        cv2.rectangle(frame, box, color=(b, g, r), thickness=2)
                        cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
                        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)
                        
                        cropped_image=self.crop_image(frame,left, top, width, height)
                        self.get_pose(cropped_image)
                cv2.imshow('Inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def run_inference(self):

        if self.args.image == '' and self.args.stream == '':
            print('[Error] Please provide a valid path for --image or --stream.')
            sys.exit(0)

        if not self.args.image == '':
            self.image_inf()

        elif not self.args.stream == '':
            self.stream_inf()

        cv2.destroyAllWindows()
    
    def get_pose(self,frame):
        pose_points=dlc_live.get_pose(frame)
    
        cv2.circle(frame,(int(pose_points[0][0]),int(pose_points[0][1])),3,(0,0,255),-1)
        cv2.circle(frame,(int(pose_points[1][0]),int(pose_points[1][1])),3,(0,255,0),-1)
        cv2.circle(frame,(int(pose_points[2][0]),int(pose_points[2][1])),3,(255,0,0),-1)
        cv2.imshow("frame",frame)

    def crop_image(self,frame,x, y, w, h):
        img_back.fill(255)
        imgCrop = img_back 
        imgCrop = frame[y:y+h, x:x+w]
        
        try:
            if imgCrop.shape[1] >x_size :
                r_x = x_size/imgCrop.shape[1]
                        
                print("--------------")
                print("prev_size :", imgCrop.shape)
                imgCrop=cv2.resize(imgCrop,None,fx=r_x,fy=r_x)
                print("new_size :", imgCrop.shape)
            if imgCrop.shape[0] > y_size:
                r_y = y_size/imgCrop.shape[0]
                    
                print("--------------")
                print("prev_size :", imgCrop.shape)
                imgCrop=cv2.resize(imgCrop,None,fx=r_y,fy=r_y)
                print("new_size :", imgCrop.shape)
                    # if (width*(0.05)<=w and height*(0.05)<=h):
            x_offset = img_back.shape[1] - imgCrop.shape[1]  
            y_offset = img_back.shape[0] - imgCrop.shape[0]  
                    
            img_back[y_offset:y_offset+imgCrop.shape[0],x_offset:x_offset + imgCrop.shape[1]] = imgCrop  
        except:
            pass
        return img_back


if __name__== '__main__':

    yolo = YOLOv4.__new__(YOLOv4)
    yolo.__init__()