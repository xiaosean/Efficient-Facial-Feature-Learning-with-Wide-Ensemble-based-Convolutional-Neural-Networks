import glob
import os
import time
import sys

import cv2
import torch

# TODO: using more fluenty code
sys.path.insert(1, './DSFD_Pytorch_Inference')
import face_detection
from controller.emotion_model import EmotionRecognition as em_model


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
    return im

def put_emotion_text(im, bboxes, emotions):
    for bbox, emotion in zip(bboxes, emotions):
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        im = cv2.putText(im, emotion, (x0, y0), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
    return im

def draw_faces_and_emotion(im, bboxes, emotions):
    print(emotions)
    for bbox, emotion in zip(bboxes, emotions):
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        im = cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)
        im = cv2.putText(im, emotion, (x0, y0), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
    return im


if __name__ == "__main__":
    device="cuda:0"
    vpaths = "./video"
    vpaths = glob.glob(os.path.join(vpaths, "*.mp4"))
    print(vpaths)
    # initialize multiple faces detection
    detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    # initialize emotion recognition
    model = em_model(device=device)
    for vpath in vpaths:
        if vpath.endswith("out.mp4"): 
            continue
        print(f"v_path ={vpath}")
        cap = cv2.VideoCapture(vpath)
        while(cap.isOpened()):
            t = time.time()
            ret, frame = cap.read()
            # results = tfnet.return_predict(frame)
            
            if ret:
                # detect face
                boxes = detector.detect(
                    frame[:, :, ::-1]
                )
                print(f"Detection face time: {time.time()- t:.3f}")
                # only using coord [x0, y0, x1, y1]
                box_coords =  boxes[:, :4]
                # if get face
                if boxes.any():
                    # tensor_img = torch.as_tensor(frame, dtype=torch.float32)                    
                    tensor_box_coords = torch.as_tensor(box_coords, dtype=torch.float32).to(device)
                    # Predict emotion from people
                    # emotions = model([tensor_img], [tensor_boxes])
                    emotions, affect, emotions_idx = model(frame, [tensor_box_coords])
                    # Draw bounding box for visiualization
                    # frame = draw_faces(frame, box_coords)
                    # frame = put_emotion_text(frame, box_coords, emotions)
                    frame = draw_faces_and_emotion(frame, box_coords, emotions)
                    

                cv2.imshow('frame', frame)
                print(f"Overall runtime each frame: {time.time()- t:.3f}")
                print('FPS {:1f}'.format(1/(time.time() -t)))
                if cv2.waitKey(1)  & 0xFF == ord('q'):
                    break
            else:
                break
                
        cap.release()
    cv2.destroyAllWindows()
