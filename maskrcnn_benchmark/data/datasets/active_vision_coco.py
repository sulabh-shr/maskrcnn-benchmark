import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from collections import OrderedDict

from maskrcnn_benchmark.structures.bounding_box import BoxList


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class ActiveVisionCOCODataset(torchvision.datasets.coco.CocoDetection):
    
    # CLASSES = {
    #  0 : 'background',
    #  1 : 'advil_liqui_gels',
    #  2 : 'aunt_jemima_original_syrup',
    #  3 : 'bumblebee_albacore',
    #  4 : 'cholula_chipotle_hot_sauce',
    #  5 : 'coca_cola_glass_bottle',
    #  6 : 'crest_complete_minty_fresh',
    #  7 : 'crystal_hot_sauce',
    #  8 : 'expo_marker_red',
    #  9 : 'hersheys_bar',
    #  10 : 'honey_bunches_of_oats_honey_roasted',
    #  11 : 'honey_bunches_of_oats_with_almonds',
    #  12 : 'hunts_sauce',
    #  13 : 'listerine_green',
    #  14 : 'mahatma_rice',
    #  15 : 'nature_valley_granola_thins_dark_chocolate',
    #  16 : 'nutrigrain_harvest_blueberry_bliss',
    #  17 : 'pepto_bismol',
    #  18 : 'pringles_bbq',
    #  19 : 'progresso_new_england_clam_chowder',
    #  20 : 'quaker_chewy_low_fat_chocolate_chunk',
    #  21 : 'red_bull',
    #  22 : 'softsoap_clear',
    #  23 : 'softsoap_gold',
    #  24 : 'softsoap_white',
    #  25 : 'spongebob_squarepants_fruit_snaks',
    #  26 : 'tapatio_hot_sauce',
    #  27 : 'vo5_tea_therapy_healthful_green_tea_smoothing_shampoo',
    #  28 : 'nature_valley_sweet_and_salty_nut_almond',
    #  29 : 'nature_valley_sweet_and_salty_nut_cashew',
    #  30 : 'nature_valley_sweet_and_salty_nut_peanut',
    #  31 : 'nature_valley_sweet_and_salty_nut_roasted_mix_nut',
    #  32 : 'paper_plate',
    #  33 : 'red_cup'
    # }

    def __init__(
        self, root, ann_file, remove_images_without_annotations=False, transforms=None, use_difficult=True
        ):
        super(ActiveVisionCOCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        
        # del self.coco.cats[0]

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        
        print(f'The categories are: {self.categories}')

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        print(f'The category id to contiguous id is :\n{self.json_category_id_to_contiguous_id}')

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx, get_filename=True):
        # load the image as a PIL Image

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        file_name = coco.loadImgs(img_id)[0]['file_name']
        folder_name = coco.dataset['img_folder_map'][file_name]
        img_path = os.path.join(self.root, folder_name, 'jpg_rgb', file_name)
        # img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path)

        anno = target
        # if self.transforms is not None:
        #     img, anno = self.transforms(img, target)

        # anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # if anno and "segmentation" in anno[0]:
        #     masks = [obj["segmentation"] for obj in anno]
        #     masks = SegmentationMask(masks, img.size, mode='poly')
        #     target.add_field("masks", masks)

        # if anno and "keypoints" in anno[0]:
        #     keypoints = [obj["keypoints"] for obj in anno]
        #     keypoints = PersonKeypoints(keypoints, img.size)
        #     target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # print(f'Index {idx:<5} : {file_name} of {folder_name:<12} has labels {classes}')
        if get_filename:
            return img, target, (idx, file_name)
        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    # def __len__(self):
    #     return 10