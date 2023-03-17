# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import math
import json, re, en_vectors_web_lg, random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as Data
from omni_res.utils.distributed import is_main_process
import omni_res.datasets.transforms.transforms as T

def make_transforms(imsize, split, mean, std, has_label=True, aug=True):
    if split == 'train' or split == 'train+val':
        # scales=[256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608]
        scales = [imsize - 32 * i for i in range(7)]
        crop_prob = 0.5
        
        if not has_label: # omni dataset
            # q: randomresize + horizontalflip + colorjitter + aug_translate | w: horizontalflip
            return T.Compose([
                T.RandomResize(scales),
                T.RandomSelect(
                    T.RandomResize(scales),
                    T.Compose([
                        T.RandomResize([400, 500, 600], with_long_side=False),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales),
                    ]),
                    p=crop_prob
                ),
                T.RandomHorizontalFlip(),
            ]), T.Compose([
                T.ColorJitter(0.4, 0.4, 0.4),
                T.GaussianBlur(aug_blur=False),
            ]), T.Compose([
                T.ToTensor(),
                T.NormalizeAndPad(mean=mean,std=std, size=imsize, aug_translate=False)
            ])
        else:
            if aug:
                return T.Compose([
                    T.RandomSelect(
                        T.RandomResize(scales),
                        T.Compose([
                            T.RandomResize([400, 500, 600], with_long_side=False),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales),
                        ]),
                        p=crop_prob
                    ),
                    T.ColorJitter(0.4, 0.4, 0.4),
                    T.GaussianBlur(aug_blur=False),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.NormalizeAndPad(mean=mean,std=std, size=imsize, aug_translate=False)
                ])
            else:
                return T.Compose([
                T.RandomResize([imsize]),
                T.ToTensor(),
                T.NormalizeAndPad(mean=mean,std=std, size=imsize),
                ])

    if split in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(mean=mean,std=std, size=imsize),
        ])

    raise ValueError(f'unknown {split}')

class RefCOCODataSet(Data.Dataset):
    def __init__(self, 
                 ann_path,
                 image_path,
                 mask_path,
                 sup_ann_path,
                 omni_ann_path,
                 input_shape,
                 flip_lr,
                 max_token_length,
                 use_glove=True, 
                 split="train", 
                 dataset="refcoco",
                 label="omni",
                 size=None,
                 mean=None,
                 std=None,
                 aug=True,
        ):
        super(RefCOCODataSet, self).__init__()
        assert dataset in ['refcoco', 'refcoco+', 'refcocog', 'referit', 'vg', 'merge', 'refcoco_merge']
        self.dataset = dataset
        self.label = None

        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        if label=='omni':
            self.label = 'omni'
            stat_refs_list=json.load(open(omni_ann_path[dataset], 'r'))
        elif split=='train':
            stat_refs_list=json.load(open(sup_ann_path[dataset], 'r'))
        else:
            stat_refs_list=json.load(open(sup_ann_path[dataset], 'r'))

        self.split=split
        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        
        for split_ in splits:
            self.refs_anno += stat_refs_list[split_]

        self.image_path = image_path[dataset]
        if dataset not in ['vg']:
            self.mask_path = mask_path[dataset]
        self.input_shape=input_shape
        self.flip_lr = flip_lr if split=='train' else False
        
        # Define run data size
        if split=='train' or split=='train+val':
            data_size = size - len(self.refs_anno)
            if len(self.refs_anno) < data_size:
                num_repeat = math.ceil(data_size / len(self.refs_anno))
                self.refs_anno = self.refs_anno*num_repeat
                self.refs_anno = random.sample(self.refs_anno, data_size)
            
        self.data_size = len(self.refs_anno)

        if is_main_process():
            print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix, self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(json.load(open(ann_path[dataset], 'r')), use_glove)
        self.token_size = self.token_to_ix.__len__()

        if is_main_process():
            print(' ========== Question token vocab size:', self.token_size)

        self.max_token = max_token_length
        if self.max_token == -1:
            self.max_token = max_token
        
        if is_main_process():
            print('Max token length:', max_token, 'Trimmed to:', self.max_token)
            print('Finished!')
            print('')

        if self.label == 'omni':
            self.transforms_w, self.transforms_q, self.transforms_t = make_transforms(input_shape[0], self.split, mean, std, has_label=False, aug=aug)
        else:
            self.transforms = make_transforms(input_shape[0], self.split, mean, std, aug=aug)


    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref = refs[np.random.choice(len(refs))]
        return ref

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['refcoco','refcoco+','refcocog','refcoco_merge']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.dataset=='vg':
            img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'].split('/')[-1])
        elif self.dataset == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
            else:
                iid=self.refs_anno[idx]['iid']
            img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
        else:
            assert NotImplementedError

        image = Image.open(img_path).convert('RGB')

        if self.dataset in ['refcoco','refcoco+','refcocog','referit']:
            mask=np.load(os.path.join(self.mask_path,'%d.npy'%self.refs_anno[idx]['mask_id']))
        elif self.dataset in ['refcoco_merge']:
            if self.split=='val':
                mask=np.load(os.path.join(os.path.join(self.mask_path,'refcoco'),'%d.npy'%self.refs_anno[idx]['mask_id']))
            else:
                mask=np.load(os.path.join(os.path.join(self.mask_path, self.refs_anno[idx]['dataset']),'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.size[0],image.size[1]], dtype=np.uint8)

        box = np.array(self.refs_anno[idx]['bbox'])
        mask = Image.fromarray(mask*255)
        return image,mask,box,self.refs_anno[idx]['mask_id'],self.refs_anno[idx]['iid']

    def __getitem__(self, idx):
        ref_iter = self.load_refs(idx)
        image_iter, mask_iter, gt_box_iter, mask_id, iid = self.load_img_feats(idx)
        
        w, h = image_iter.size
        input_dict = {'img':image_iter, 'box':box_xywh_to_xyxy(torch.from_numpy(gt_box_iter).float()), 'mask':mask_iter, 'text':ref_iter}

        percent = random.random()
        if self.label=='omni':
            input_dict_w, _ = self.transforms_w(copy.deepcopy(input_dict), percent)   # weak augmentation
            input_dict_q, _ = self.transforms_q(copy.deepcopy(input_dict_w), percent) # strong augmentation
            ori_img = input_dict_q['img']
            ori_box = input_dict_q['box'].clone()

            input_dict_w, _ = self.transforms_t(input_dict_w, percent)
            input_dict_q, _ = self.transforms_t(input_dict_q, percent)
            
            while input_dict_q['box'][:,0]>=1 or input_dict_q['box'][:,1]>=1 or input_dict_q['box'][:,2]==0 or input_dict_q['box'][:,3]==0:
                if gt_box_iter[0]>w or gt_box_iter[1]>h:
                    print('gt box outside image')
                    break
                input_dict_w, _ = self.transforms_w(copy.deepcopy(input_dict), percent)   # weak augmentation
                input_dict_q, _ = self.transforms_q(copy.deepcopy(input_dict_w), percent) # strong augmentation
                ori_img = input_dict_q['img']
                ori_box = input_dict_q['box'].clone()
                input_dict_w, _ = self.transforms_t(input_dict_w, percent)
                input_dict_q, _ = self.transforms_t(input_dict_q, percent)

            ref_iter_w = self.proc_ref(input_dict_w['text'],self.token_to_ix,self.max_token)
            ref_iter_q = self.proc_ref(input_dict_q['text'],self.token_to_ix,self.max_token)

            info_iter_w = {'h':h, 'w':w, 'nh': input_dict_w['info_img'][0], 'nw': input_dict_w['info_img'][1], 'top':input_dict_w['info_img'][3], 'left': input_dict_w['info_img'][2],
                           'flip': True if 'flip' in input_dict_w.keys() else False,'crop': input_dict_w['crop'] if 'crop' in input_dict_w.keys() else None}
            info_iter_q = {'h':h, 'w':w, 'nh': input_dict_q['info_img'][0], 'nw': input_dict_q['info_img'][1], 'top':input_dict_q['info_img'][3], 'left': input_dict_q['info_img'][2],
                           'flip': True if 'flip' in input_dict_q.keys() else False,'crop': input_dict_q['crop'] if 'crop' in input_dict_q.keys() else None}

            return \
                ori_img, \
                ori_box, \
                ref_iter, \
                torch.from_numpy(ref_iter_w).long(), \
                input_dict_w['img'], \
                torch.from_numpy(ref_iter_q).long(), \
                input_dict_q['img'], \
                input_dict_q['mask'], \
                input_dict_q['box'], \
                info_iter_q
        else:
            input_dict_process, percent = self.transforms(copy.deepcopy(input_dict), percent)
            while input_dict_process['box'][:,0]>=1 or input_dict_process['box'][:,1]>=1 or input_dict_process['box'][:,2]==0 or input_dict_process['box'][:,3]==0:
                input_dict_process, percent = self.transforms(copy.deepcopy(input_dict), percent)

            ref_iter = self.proc_ref(input_dict_process['text'], self.token_to_ix, self.max_token)
            info_iter = [h,w,*input_dict_process['info_img'], iid]

            return \
                torch.from_numpy(ref_iter).long(), \
                input_dict_process['img'], \
                input_dict_process['mask'], \
                input_dict_process['box'], \
                mask_id, \
                np.array(info_iter)


    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [x, y, x + w, y + h]
    return torch.stack(b, dim=-1)