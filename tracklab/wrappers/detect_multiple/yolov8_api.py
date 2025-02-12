import os
import torch
import pandas as pd

from typing import Any
from tracklab.pipeline.imagelevel_module import ImageLevelModule

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from tracklab.utils.coordinates import ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv8(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        if not os.path.exists(cfg.path_to_checkpoint_player):
            log.error(f"Checkpoint path {cfg.path_to_checkpoint_player} does not exist.\n\n")
        if not os.path.exists(cfg.path_to_checkpoint_ball):
            log.error(f"Checkpoint path {cfg.path_to_checkpoint_ball} does not exist.\n\n")    
        
        # Initilize models
        self.player_model = YOLO(cfg.path_to_checkpoint_player)
        self.player_model.to(device)
        
        self.ball_model = YOLO(cfg.path_to_checkpoint_ball)
        self.ball_model.to(device)
        
        self.id = 0

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):

        images, shapes = batch
        player_results_by_image = self.player_model(
            images,
            iou=0.5,
            imgsz=1280,
        )
        ball_results_by_image = self.ball_model(
            images,
            iou=0.2,
            imgsz=1280,
        )
            
        print(f"\n\nNumber of player predictions: {len(player_results_by_image[0])}")
        print(f"Number of ball predictions: {len(ball_results_by_image[0])}")
        
        detections = []
        for results, shape, (_, metadata) in zip(
            player_results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence_player:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
                    
        for results, shape, (_, metadata) in zip(
            ball_results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # print(f"Detected ball ({bbox.cls}) with confidence {bbox.conf}")
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence_ball:
                        detections.append(
                            pd.Series(
                                dict(
                                    image_id=metadata.name,
                                    bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                    bbox_conf=bbox.conf[0],
                                    video_id=metadata.video_id,
                                    category_id=4, # 'ball' class in posetrack
                                ),
                                name=self.id,
                            )
                        )
                        self.id += 1
            
        return detections
