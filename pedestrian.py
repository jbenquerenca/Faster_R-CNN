import json, os
from fvcore.common.file_io import PathManager
from collections import defaultdict
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes, BoxMode
def load_instances(dirname: str, split: str):
    with PathManager.open(os.path.join(dirname, "Annotations", split + ".json")) as f: annot_data = json.load(f)
    annots_dict = defaultdict(list)
    imgs_dict = {img["id"]:img for img in annot_data["images"]}
    for anno in annot_data["annotations"]: annots_dict[anno["image_id"]].append(anno)
    dicts = list()
    for img_id, annos in annots_dict.items():
        if not "Caltech" in dirname: img_info = imgs_dict[str(img_id)] if split=="train" else imgs_dict[img_id]
        else: img_info = imgs_dict[img_id]
        r = dict(file_name=os.path.join(dirname, "Images", img_info["file_name"]), image_id=img_id, height=img_info["height"], width=img_info["width"])
        instances = list()
        for anno in annos:
            bbox = BoxMode.convert([round(float(x), 3) for x in anno["bbox"]], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            instance = dict(bbox=bbox, area=Boxes([bbox]).area()[0].item(),
                iscrowd=anno["iscrowd"], ignore=anno["iscrowd"] if "ignore" not in anno else anno["ignore"],
                category_id=anno["category_id"], bbox_mode=BoxMode.XYXY_ABS)
            instances.append(instance)
        r["annotations"] = instances
        dicts.append(r)
    return dicts
def register_dataset(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=["_background", "pedestrian"], dirname=dirname, split=split)
def register():
    root = os.getenv("DETECTRON2_DATASETS", "datasets")
    SPLITS = [
        ("tju-pedestrian-traffic_train", "TJU-Pedestrian-Traffic", "train"),
        ("tju-pedestrian-traffic_val", "TJU-Pedestrian-Traffic", "val"),
        ("tju-pedestrian-traffic_test", "TJU-Pedestrian-Traffic", "test"),
        ("caltech_pedestrians_train", "Caltech_Pedestrians", "train"),
        ("caltech_pedestrians_val", "Caltech_Pedestrians", "val"),
        ("caltech_pedestrians_test", "Caltech_Pedestrians", "test")]
    for name, dirname, split in SPLITS:
        register_dataset(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "coco"