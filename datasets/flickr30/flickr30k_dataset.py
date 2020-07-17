import glob
import os

import nltk
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive, download_url

from datasets.flickr30.flickr30k_entities_utils import get_sentence_data
from flickr30k_settings import dataset_dir

FLICKR30K_URL = "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k-images.tar"
FLICKR30K_GRAPH_URL = "wget http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz"
ANNOTATIONS_URL = "https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip"
SPLITS_URL_FMT = "https://github.com/BryanPlummer/flickr30k_entities/raw/master/{}.txt"
SPLITS = ('test', 'train', 'val')

IMAGES_DIR = 'flickr30k-images'
PHRASE_TYPES_ALL = ('scene', 'people', 'other', 'notvisual', 'clothing', 'bodyparts',
                    'animals', 'vehicles', 'instruments')
PHRASE_TYPES_DEFAULTS = ('people', 'scene', 'animals', 'vehicles', 'instruments')
ALL_IMGS_NUM = 31783
FLICKR30K_CPI = 5


class Flickr30kDataset(data.Dataset):
    root_dir = None
    img_sent_ptype, img_ptype = None, None

    @classmethod
    def init(cls, root_dir=dataset_dir, force_rebuild_cache=False):
        print("\n" + "*" * 80 + "\n Flickr30k DATASET INIT\n" + "*" * 80)
        cls.root_dir = root_dir
        cls.force_rebuild_cache = force_rebuild_cache

        cls._download_images()
        cls._download_splits()
        cls._download_annotations()

        cls._prepare_sentences(force_rebuild_cache)
        cls.img_sent_ptype, cls.img_ptype = torch.load(cls._preproc_file())
        assert len(cls.img_sent_ptype) == ALL_IMGS_NUM
        assert len(cls.img_ptype) == ALL_IMGS_NUM

    @classmethod
    def _download_images(cls):
        if not os.path.exists(f'{cls.root_dir}/{IMAGES_DIR}'):
            print('Downloading Flickr30k...')
            download_and_extract_archive(FLICKR30K_URL, cls.root_dir, cls.root_dir, 'flickr30k-images.tar')

    @classmethod
    def _download_splits(cls):
        for split in SPLITS:
            if not os.path.exists(f'{cls.root_dir}/{split}.txt'):
                print('Downloading split: {split}')
                download_url(SPLITS_URL_FMT.format(split), cls.root_dir, f'{split}.txt')

    @classmethod
    def _download_annotations(cls):
        if not os.path.exists(f'{cls.root_dir}/Sentences'):
            print('Downloading Flickr30k Annotations ...')
            download_and_extract_archive(ANNOTATIONS_URL, cls.root_dir, cls.root_dir, 'annotations.zip')

    @classmethod
    def _preproc_file(cls):
        return f'{cls.root_dir}/preproc_sent_pt.pt'

    @classmethod
    def is_init(cls):
        return cls.img_sent_ptype is not None and cls.img_ptype is not None

    @classmethod
    def _split_file(cls, split):
        return f'{cls.root_dir}/{split}.txt'

    @classmethod
    def read_ids(cls, split):
        with open(cls._split_file(split), 'rt') as f:
            return [int(l.strip()) for l in f.readlines()]

    @classmethod
    def _prepare_sentences(cls, force_rebuild_cache):
        if force_rebuild_cache or (not os.path.exists(cls._preproc_file())):
            img_sent_ptype = {}
            img_ptype = {}
            for f in tqdm.tqdm(glob.glob(f'{cls.root_dir}/Sentences/*.txt'),
                               'Reading sentences data'):
                a = get_sentence_data(f)
                assert len(a) == FLICKR30K_CPI
                _id = int(f.split('/')[-1].split('.')[0])
                _s = []
                _pt = []
                _set = set()
                for s in a:
                    pt = []
                    for p in s['phrases']:
                        for _p in p['phrase_type']:
                            pt.append(_p)
                            _set.add(_p)
                    _s.append(s['sentence'])
                    _pt.append(pt)
                img_sent_ptype[_id] = (_s, _pt)
                img_ptype[_id] = _set
            torch.save((img_sent_ptype, img_ptype), cls._preproc_file())

    """Flicker30k Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, split=None, caption_vocab=None, h5_feats=None, img_ids=None, transform=None, multicaption=False,
                 train=True, # just for compatibility with COCO Dataset
                 cpi=FLICKR30K_CPI, use_extra=False):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            split: one of the split train/val/test.
            caption_vocab: vocabulary wrapper.
            h5_feats: pre-extracted features
            img_ids: sub-set of images ids
            transform: image transformer.
            multicaption:
            cpi:
            use_extra:
        """
        assert split in SPLITS, f'Wrong split: {split}. Should be one of: {SPLITS}'
        # if train:
        #     assert split == 'train'
        assert Flickr30kDataset.is_init(), "You have to initialize Flickr30k with init classmethod"
        if caption_vocab is None:
            multicaption = True  # using multicaption we cycle over images not over captions.
            cpi = 1

        self.cpi = cpi

        # assert (split is not None) ^ (img_ids is not None), 'Provide split or ids (but not both)!'

        if img_ids is None:
            self.img_ids = Flickr30kDataset.read_ids(split)
        else:
            self.img_ids = img_ids

        self.vocab = caption_vocab
        self.transform = transform
        self.multicaption = multicaption
        self.use_extra = use_extra
        self.h5 = None
        self.h5_filenames = None
        if h5_feats:
            self.h5 = h5_feats
            self.h5_feats = torch.tensor(np.array(self.h5['resnet152']))
            h5_img_ids = np.array(self.h5['image_ids'])
            flickr30k_id_to_idx = {k: v for k, v in zip(h5_img_ids.tolist(), range(len(h5_img_ids)))}
            self.h5_feats_flickr30k_idx = {k: self.h5_feats[v] for k, v in flickr30k_id_to_idx.items()}

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        if self.multicaption:
            img_id = self.img_ids[index]
            captions, _ = Flickr30kDataset.img_sent_ptype[img_id]
        else:
            img_id = self.img_ids[index // self.cpi]
            captions, _ = Flickr30kDataset.img_sent_ptype[img_id]
            captions = [captions[index % self.cpi]]

        path = os.path.join('flickr30k-images', f'{img_id}.jpg')
        if self.h5 is not None:
            image = self.h5_feats_flickr30k_idx[img_id]
        else:
            image = Image.open(os.path.join(Flickr30kDataset.root_dir, path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

        extra = None if not self.use_extra else {'image_id': img_id}

        if self.vocab is not None:
            # Convert caption (string) to word ids.
            targets = []
            for caption in captions:
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                caption = []
                caption.append(self.vocab('<start>'))
                caption.extend([self.vocab(token) for token in tokens if token != '.'])
                caption.append(self.vocab('<end>'))
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
            return len(self.img_ids) * self.cpi

    def get_loader(self, batch_size, shuffle, num_workers, pin_memory=True, drop_last=False, sampler=None,
                   batch_sampler=None):
        # Data loader for COCO dataset, t his will return (images, captions, lengths) for each iteration.
        # images: a tensor of shape (batch_size, 3, 224, 224).
        # captions: a tensor of shape (batch_size, padded_length).
        # lengths: a list indicating valid length for each caption. length is (batch_size).
        return DataLoader(self, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                          drop_last)


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
