import json
import os

from collections import defaultdict
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# some images in tju throw an exception loading in PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image_info_and_annotations(annotation_path):
    annotations = defaultdict(dict)

    with open(annotation_path) as f:
        annotations_json = json.load(f)

    for img in annotations_json["images"]: 
        annotations[img["id"]] = {
            "img_info": img,
            "instances": list()
        }
    
    for anno in annotations_json["annotations"]: 
        annotations[anno["image_id"]]["instances"].append(anno)

    return annotations

def load_pedestrian_instances(dirname: str, split: str):
    return [{
            "file_name": os.path.join(dirname, "images", img_dict["img_info"]["file_name"]),
            "image_id": img_dict["img_info"]["id"],
            "height": img_dict["img_info"]["height"],
            "width": img_dict["img_info"]["width"],
            "annotations": [{
                    "category_id": 1,
                    "bbox": isinstance["bbox"],
                    "bbox_mode": BoxMode.XYWH_ABS 
                } for instance in img_dict["instances"] if ("ignore" in instance and not instance["ignore"]) or (not instance["iscrowd"])]
            } for img_dict in read_image_info_and_annotations(os.path.join(dirname, "annotations", f"{split}.json"))]

def register_pedestrian_dataset(name, dirname, split):
    DatasetCatalog.register(name, 
                            lambda: load_pedestrian_instances(dirname, split))

    MetadataCatalog.get(name).set(
        thing_classes=["_background", "pedestrian"], 
        dirname=dirname, 
        split=split)

def register_all_pedestrian_datasets(root):
    SPLITS = [
        ("caltech_pedestrians_train",    "Caltech_Pedestrians",    "train"),
        ("caltech_pedestrians_val",      "Caltech_Pedestrians",    "val"),
        ("caltech_pedestrians_test",     "Caltech_Pedestrians",    "test"),
        ("tju-pedestrian-traffic_train", "TJU-Pedestrian-Traffic", "train"),
        ("tju-pedestrian-traffic_val",   "TJU-Pedestrian-Traffic", "val"),
        ("tju-pedestrian-traffic_test",  "TJU-Pedestrian-Traffic", "test"),
    ]
    for name, dirname, split in SPLITS:
        register_pedestrian_dataset(name, os.path.join(root, dirname), split),
        MetadataCatalog.get(name).evaluator_type = "pedestrian"