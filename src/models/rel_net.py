# -*- coding: utf-8 -*-
"""A simple net fusing multiple features."""

from time import time

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from src.utils.train_test_utils import SGGTrainTester
from src.tools.early_stopping_scheduler import EarlyStopping
from src.models.object_classifier import ObjectClassifier


class RelNet(nn.Module):
    """Extends PyTorch nn.Module."""

    def __init__(self, num_classes, train_top=False):
        """Initialize layers."""
        super().__init__()

        # Visual features
        self.fc_subject_v = nn.Sequential(nn.Linear(2048, 256), nn.ReLU())
        self.fc_predicate_v = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())
        self.fc_object_v = nn.Sequential(nn.Linear(2048, 256), nn.ReLU())
        self.fc_visual = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())

        # Spatial-language features
        self.fc_subject_l = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object_l = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 8), nn.ReLU())
        self.fc_delta = nn.Sequential(
            nn.Linear(38, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.fc_spatial_lang = nn.Sequential(nn.Linear(768, 256), nn.ReLU())

        # Fusion
        self.fc_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes))
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2))
        self.feat_extractor = ObjectClassifier(num_classes, train_top, False)
        self.softmax = nn.Softmax(dim=1)
        self.mode = 'train'

    def forward(self, subj_feats, pred_feats, obj_feats, deltas,
                subj_masks, obj_masks, subj_embs, obj_embs):
        """Forward pass, return output scores."""
        v_features = self.fc_visual(torch.cat((
            self.fc_subject_v(subj_feats),
            self.fc_predicate_v(pred_feats),
            self.fc_object_v(obj_feats)
        ), dim=1))
        sl_features = self.fc_spatial_lang(torch.cat((
            self.fc_subject_l(subj_embs),
            self.mask_net(
                torch.cat((subj_masks, obj_masks), dim=1)).view(-1, 128),
            self.fc_delta(deltas),
            self.fc_object_l(obj_embs)
        ), dim=1))
        features = torch.cat((v_features, sl_features), dim=1)
        scores = self.fc_classifier(features)
        scores_bin = self.fc_classifier_bin(features)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin

    def feat_forward(self, base_features, im_info, rois):
        """Forward of feature extractor."""
        _, top_features, _, _, _ = self.feat_extractor(
            base_features, im_info, rois)
        return top_features


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features):
        """Initialize instance."""
        super().__init__(net, config, features)
        self._obj2vec = config.obj2vec
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        if self._task == 'sgcls':
            self.obj_classifier = ObjectClassifier(
                config.num_obj_classes, False, False)
            checkpoint = torch.load(
                self._models_path
                + 'object_classifier_objcls_' + self._dataset + '.pt')
            self.obj_classifier.load_state_dict(checkpoint['model_state_dict'])
            if self._use_cuda:
                self.obj_classifier.cuda()
            else:
                self.obj_classifier.cpu()

    def _net_forward(self, batch, step):
        relations = self.data_loader.get('relations', batch, step)
        base_features = self.data_loader.get('base_features', batch, step)
        if self.net.mode == 'train' or self._task != 'sgcls':
            obj_vecs = self.data_loader.get('object_1hot_vectors', batch, step)
        elif self._task == 'sgcls':
            _, _, obj_vecs, _, _ = self.obj_classifier(
                base_features,
                self.data_loader.get('image_info', batch, step),
                self.data_loader.get('object_rcnn_rois', batch, step))
        obj_features = self.net.feat_forward(
            base_features,
            self.data_loader.get('image_info', batch, step),
            self.data_loader.get('object_rcnn_rois', batch, step))
        obj_features = obj_features.mean(3).mean(2)
        embeddings = self._obj2vec[obj_vecs.argmax(1)]
        deltas = self.data_loader.get('box_deltas', batch, step)
        masks = self.data_loader.get('object_masks', batch, step)
        pred_rois = self.data_loader.get('predicate_rcnn_rois', batch, step)
        if self.net.mode == 'train':
            deltas = deltas[:44 ** 2]
            pred_rois = pred_rois[:, :44 ** 2, :]
            relations = relations[:44 ** 2]
        rel_batch_size = 128
        scores, subject_scores, object_scores = [], [], []
        bin_scores = []
        for btch in range(1 + (len(relations) - 1) // rel_batch_size):
            rels = relations[btch * rel_batch_size:(btch + 1) * rel_batch_size]
            pred_features = self.net.feat_forward(
                base_features,
                self.data_loader.get('image_info', batch, step),
                pred_rois[
                    :, btch * rel_batch_size:(btch + 1) * rel_batch_size, :]
            ).mean(3).mean(2)
            b_scores, b_bin_scores = self.net(
                torch.stack([obj_features[r[0]] for r in rels], dim=0),
                pred_features,
                torch.stack([obj_features[r[1]] for r in rels], dim=0),
                deltas[btch * rel_batch_size:(btch + 1) * rel_batch_size],
                torch.stack([masks[r[0]] for r in rels], dim=0),
                torch.stack([masks[r[1]] for r in rels], dim=0),
                torch.stack([embeddings[r[0]] for r in rels], dim=0),
                torch.stack([embeddings[r[1]] for r in rels], dim=0)
            )
            scores.append(b_scores)
            bin_scores.append(b_bin_scores)
            subject_scores.append(
                torch.stack([obj_vecs[r[0]] for r in rels], dim=0))
            object_scores.append(
                torch.stack([obj_vecs[r[1]] for r in rels], dim=0))
        return (
            torch.cat(scores, dim=0),
            torch.cat(bin_scores, dim=0),
            torch.cat(subject_scores, dim=0), torch.cat(object_scores, dim=0)
        )

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        scores, bin_scores, _, _ = self._net_forward(batch, step)
        targets = self.data_loader.get('predicate_targets', batch, step)
        bg_targets = self.data_loader.get('bg_targets', batch, step)
        if len(bg_targets[bg_targets == 1]) >= 1:
            scale = len(bg_targets) / len(bg_targets[bg_targets == 1])
        else:
            scale = 1
        loss = self.criterion(scores, targets) * scale
        loss = loss + self.cross_entropy(bin_scores, bg_targets)
        return loss

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        scores, bin_scores, s_scores, o_scores = self._net_forward(batch, step)
        return scores * bin_scores[:, 1].unsqueeze(-1), s_scores, o_scores


def train_test(config):
    """Train and test a net."""
    config.logger.debug(
        'Tackling %s for %d classes' % (config.task, config.num_classes))
    net = RelNet(num_classes=config.num_classes, train_top=config.train_top)
    features = {
        'bg_targets',
        'box_deltas',
        'images',
        'image_info',
        'object_1hot_vectors',
        'object_masks',
        'object_rcnn_rois',
        'predicate_rcnn_rois',
        'predicate_targets',
        'relations'
    }
    if config.task not in {'preddet', 'predcls'}:
        features = features.union({'images', 'image_info', 'object_rcnn_rois'})
    train_tester = TrainTester(net, config, features)
    top_net_params = [
        param for name, param in net.named_parameters()
        if 'top_net' in name and param.requires_grad]
    other_net_params = [
        param for name, param in net.named_parameters()
        if 'top_net' not in name and param.requires_grad]
    optimizer = optim.Adam(
        [
            {'params': top_net_params, 'lr': 0.0002},
            {'params': other_net_params}
        ], lr=0.002, weight_decay=config.weight_decay)
    if config.use_early_stopping:
        scheduler = EarlyStopping(
            optimizer, factor=0.3, patience=1, max_decays=2)
    else:
        scheduler = MultiStepLR(optimizer, [4, 8], gamma=0.3)
    t_start = time()
    train_tester.train(
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(
            ignore_index=config.num_classes - 1, reduction='none'),
        scheduler=scheduler,
        epochs=30 if config.use_early_stopping else 10)
    config.logger.info('Training time: ' + str(time() - t_start))
    train_tester.net.mode = 'test'
    t_start = time()
    train_tester.test()
    config.logger.info('Test time: ' + str(time() - t_start))

