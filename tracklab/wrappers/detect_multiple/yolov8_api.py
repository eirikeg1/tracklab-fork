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
        if not os.path.exists(cfg.path_to_checkpoint):
            log.error(f"Checkpoint path {cfg.path_to_checkpoint} does not exist.\n\n")
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
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
        results_by_image = self.model(images)
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # check for `person` class
                # print(f"Detected {bbox.cls} with confidence {bbox.conf}")
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
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
                elif bbox.cls == 1 and bbox.conf >= self.cfg.min_confidence:
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
                elif not bbox.cls in [0, 1]:
                    print(f"Detected {bbox.cls} ({type(bbox.cls)}) with confidence {bbox.conf}")
                    
        return detections
