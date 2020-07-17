from os.path import join

import h5py
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader


class CocoDataset(data.Dataset):
    coco_dir = None
    coco_train = None
    coco_val = None

    @classmethod
    def init(cls, coco_root_dir):
        print("\n" + "*"*80 + "\n COCODATASET INIT\n" + "*"*80)
        print("Init coco training-set")
        cls.coco_train = COCO(join(coco_root_dir, 'annotations/captions_train2014.json'))
        print("\nInit coco validation-set")
        cls.coco_val = COCO(join(coco_root_dir, 'annotations/captions_val2014.json'))
        print("*"*80 + "\nDone!\n" + "*" * 80 + "\n")
        cls.coco_dir = coco_root_dir

    @classmethod
    def is_init(cls):
        return cls.coco_train is not None and cls.coco_val is not None

    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, img_dir, caption_vocab=None, h5_feats=None, img_ids=None, transform=None, multicaption=False,
                 train=True, cpi=5, use_extra=True):
        """
        COCO Dataset for a single training task.

        Args:
            img_dir: coco dataset image directory path
            caption_vocab: vocabulary of current task
            h5_feats: image features pre-extracted with a cnn. If None, only the e
            img_ids: indexes of the examples in the original coco-dataset. If None, all the examples will be used,
                        otherwise only the selected one will be used.
            transform: transform function applied to the images (not used if h5_feats is provided).
            multicaption: if True the dataloader will cycle over the example indices and will return all the captions
                            for each example, otherwise will cycle over the captions and will return pairs of
                            (image/feature, caption).
            train: select COCO training set (when True) or COCO validation set split (when False).
            cpi: number of captions per image for the dataset (by default, coco has 5 cpi).
            use_extra: if True the dataloader will return an extra tuple containing the file name, the image id and
                        the annotation id for each example.
        """
        assert CocoDataset.is_init(), "You have to initialize CocoDataset with init classmethod"
        if caption_vocab is None:
            multicaption = True  # using multicaption we cycle over images not over captions.
            cpi = 1

        self.coco = CocoDataset.coco_train if train else CocoDataset.coco_val
        self.cpi = cpi
        self.img_dir = img_dir
        self.ann_ids = sorted(self.coco.anns.keys())
        self.img_ids = sorted(self.coco.imgs.keys())
        if img_ids is not None:
            self.img_ids = sorted(img_ids)
            self.ann_ids = sorted(self.coco.getAnnIds(imgIds=img_ids))
        self.vocab = caption_vocab
        self.transform = transform
        self.multicaption = multicaption
        self.use_extra = use_extra
        self.h5 = None
        self.h5_filenames = None
        if h5_feats is not None:
            self.h5 = h5_feats
            self.h5_feats = torch.tensor(np.array(self.h5['resnet152']))
            self.h5_filenames = np.array(self.h5['filenames'])
            h5_img_ids = np.array(self.h5['image_ids'])
            coco_to_h5_idx = {k: v for k, v in zip(h5_img_ids.tolist(), range(len(h5_img_ids)))}
            self.h5_feats_coco_idx = {k: self.h5_feats[v] for k, v in coco_to_h5_idx.items()}
            self.h5_filenames_coco_idx = {k: self.h5_filenames[v] for k, v in coco_to_h5_idx.items()}

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab

        if self.multicaption:
            img_id = self.img_ids[index]
            ann_ids = sorted(self.coco.getAnnIds(imgIds=[img_id]))[:self.cpi]
        else:
            ann_ids = [self.ann_ids[index]]
            img_id = self.coco.anns[ann_ids[0]]['image_id']

        captions = [self.coco.anns[ann_id]['caption'] for ann_id in ann_ids]
        if self.h5 is not None:
            image = self.h5_feats_coco_idx[img_id]
            path = self.h5_filenames_coco_idx[img_id]
        else:
            path = self.coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

        extra = None if not self.use_extra else {'file_name': path, 'image_id': img_id, 'ann_ids': ann_ids}

        if vocab is not None:
            # Convert caption (string) to word ids.
            targets = []
            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                targets.append(caption)

            from torch.nn.utils.rnn import pad_sequence
            if self.multicaption:
                lens = torch.tensor([len(t) for t in targets])
                targets = pad_sequence([torch.tensor(t) for t in targets], batch_first=True)
            else:
                targets = torch.tensor(targets[0])
                lens = torch.tensor([len(targets)])
            return image, targets, lens, extra
        else:
            return image, extra


    def __len__(self):
        if self.multicaption:
            return len(self.img_ids)
        else:
            return len(self.ann_ids)


    def get_loader(self, batch_size, shuffle, num_workers, pin_memory=True, drop_last=False, sampler=None, batch_sampler=None):
        # Data loader for COCO dataset, t his will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
        return DataLoader(self, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last)



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    if len(data[0]) == 4:
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, lens, extra = zip(*data)
    elif len(data[0]) == 2:
        images, extra = zip(*data)
        captions = lens = None

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    if captions is not None:
        # If we have multi-captions per each example:
        if len(captions[0].shape) == 2:
            cpi = captions[0].shape[0]
            lengths = torch.stack(lens)
            targets = torch.zeros(len(captions), cpi, lengths.max()).long()

            for i, caps in enumerate(captions):
                for j, cap in enumerate(caps):
                    end = lengths[i][j]
                    targets[i, j, :end] = cap[:end]

        # If we have a single caption per each example:
        else:
            lengths = torch.tensor([len(cap) for cap in captions])
            targets = torch.zeros(len(captions), max(lengths)).long()
            for i, cap in enumerate(captions):
                end = lengths[i]
                targets[i, :end] = cap[:end]

        return images, targets, lengths, extra

    else:
        return images, extra



