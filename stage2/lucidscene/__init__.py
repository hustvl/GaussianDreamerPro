#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from lucidutils.system_utils import searchForMaxIteration
from lucidscene.dataset_readers import sceneLoadTypeCallbacks,GenerateRandomCameras,GeneratePurnCameras,GenerateCircleCameras,GenerateOutCameras
# from lucidscene.gaussian_model import GaussianModel
from arguments import ModelParams, GenerateCamParams
from lucidutils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_RcamInfos

class Scene:


    def __init__(self, args : ModelParams, pose_args : GenerateCamParams, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.loaded_iter = None
        self.resolution_scales = resolution_scales
        self.pose_args = pose_args
        self.args = args


    def getRandTrainCameras(self, scale=1.0):
        rand_train_cameras = GenerateRandomCameras(self.pose_args, self.args.batch, SSAA=True)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args, SSAA=True)        
        return train_cameras[scale]


    def getPurnTrainCameras(self, scale=1.0):
        rand_train_cameras = GeneratePurnCameras(self.pose_args)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args)        
        return train_cameras[scale]


    def getCircleVideoCameras(self, scale=1.0,batch_size=120, render45 = True):
        video_circle_cameras = GenerateCircleCameras(self.pose_args,batch_size,render45)
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(video_circle_cameras, resolution_scale, self.pose_args)        
        return video_cameras[scale]