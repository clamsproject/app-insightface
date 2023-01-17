import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import sys
import time
import json
import os

def infer_on_video(file_name, threshold=1.54, update=False, tta=False, show_score=False, begin=0, duration=0, save_name=None, verbose=False):
    
    #start_time = time.time()  
    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
        
    #cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    cap = cv2.VideoCapture(str(file_name))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, begin * 1000)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    int_fps = round(fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if(save_name != None):
        video_writer = cv2.VideoWriter(str(save_name), cv2.VideoWriter_fourcc(*'FMP4'), int(fps), (width, height))

    bbox_list = []
    i = 0   
    
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:
            if(i % int_fps == 0):  
                #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                try:
                    bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)
                except:
                    bboxes = []
                    faces = []
                if len(bboxes) == 0:
                    pass
                else:
                    bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1] # personal choice   
                    results, score = learner.infer(conf, faces, targets, True)
                    for idx,bbox in enumerate(bboxes):
                        if show_score:
                            frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                        else:
                            frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                bbox_tuples = [(names[results[idx] + 1], [int(position) for position in bbox]) for idx,bbox in enumerate(bboxes)]
                bbox_list.append(bbox_tuples)
                if(save_name != None):
                    video_writer.write(frame)
        else:
            break
        i += 1
        
        if verbose and (i % (60*int_fps) == 0):
            print('{} minute processed'.format(i // (60*int_fps)))
         
        if duration != 0 and i > int_fps * duration:
            break

    last_frame = (i // int_fps) * int_fps
    bbox_per_frame = {(i*int_fps): bbox_list[i] for i in range(len(bbox_list)) if len(bbox_list[i])>0}
    cap.release()
    json_dict = {'app':'InsightFace_Pytorch','video_name':str(file_name), 'fps':fps, 'width':width, 'height':height, 'frame_count':frame_count,
                 'last_frame':last_frame, 'bounding_boxes_per_frame_index': bbox_per_frame}
    if(save_name != None):
        video_writer.release()
    return json_dict
    
