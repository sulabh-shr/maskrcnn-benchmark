# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import os
import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, filenames=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        # --------------------- Self-supervised Additions ---------------------
        # print(proposals[0].get_field("objectness"))

        rpn_filenames = []
        classifier_filenames = []

        for filename in filenames:
        	rpn_filenames.append(filename.split('.')[0]+'_rpn.pt')
        	classifier_filenames.append(filename.split('.')[0]+'_classifier.pt')
        print(f'Filenames: {filenames} | Image size: {images.tensors.shape}')
        
        base_output_dir = '/home/sulabh/workspace-ubuntu/self_supervised_ouputs/proposals'
        config = None
        rpn_dir = 'rpn'
        classifier_dir = 'classifier'

        if config is None:
        	raise ValueError("Config name must be set to save output in appropriate file")

        # Save RPN proposals
        for proposal, rpn_filename in zip(proposals, rpn_filenames):
        	torch.save(proposals, os.path.join(base_output_dir, config, rpn_dir, rpn_filename))

        # --------------------- Self-supervised Additions end -----------------

        if self.roi_heads:
            # Feature fed to final classifer, Result of Classifier, Loss
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
            print('Feature size', x.shape)
            
        	# --------------------- Self-supervised Additions -----------------
            # Save Classifier proposals
            for res, classifier_filename in zip(result, classifier_filenames):
            	torch.save(result, os.path.join(base_output_dir, config, classifier_dir, classifier_filenames))
            
            # print(f'Inferenced on {len(result)} images')
            # print(result[0].get_field("scores"))
            # print(result[0].get_field("labels"))
            # print(result[0].bbox)
        	
        	# --------------------- Self-supervised Additions end -------------

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
