from collections import Counter, defaultdict
from random import random
from typing import Dict, List

import nltk
from tqdm import tqdm

from CNIC.task import Task, Job, JobFactory
from CNIC.vocab import Vocabulary
from datasets.flickr30.flickr30k_dataset import SPLITS
from datasets.flickr30.flickr30k_dataset import Flickr30kDataset


def tasks_img_ids(split, tasks: Dict[str, List['str']]):
    """
    Disjoint with next labels splitting strategy.
    :param split: train/val/test
    :param tasks:
    :return:
    """
    dataset_split_ids = Flickr30kDataset.read_ids(split)
    phrase_types = [pt for t_name, pts in tasks.items() for pt in pts]
    print(f'Flickr30k Split: {split} - Disjoint with next labels split: \n{phrase_types}')
    i = 0
    for t_name, pts in tasks.items():
        tasks_ids = []
        for pt in pts:
            l = phrase_types[i]
            assert l == pt
            next_pt = set(phrase_types[i + 1:])
            print('Current label: ', l)
            print('Excluded: ', next_pt)
            for _id in dataset_split_ids:
                pts = Flickr30kDataset.img_ptype[_id]
                if l in pts and len(set(pts).intersection(next_pt)) == 0:
                    tasks_ids.append(_id)
            i += 1
            yield t_name, tasks_ids


def split_tasks_img_ids(job_tasks: Dict[str, List['str']], root_dir, max_samples, shuffle=False):
    Flickr30kDataset.init(root_dir)
    print("\nComputing disjoint set with next labels for Flickr30k dataset...")
    tasks_splits_img_ids = defaultdict(dict)
    for split in SPLITS:
        for tname, ids in tasks_img_ids(split, job_tasks):
            print(f'{split} label: {tname} - iamges num.: {len(ids)}')
            if split in max_samples and len(ids) > max_samples[split]:
                print(f'Too much images. Subsumpling to: {max_samples[split]}')
                if shuffle:
                    random.shuffle(ids)
                ids = ids[:max_samples[split]]
            tasks_splits_img_ids[tname][split] = ids
    return tasks_splits_img_ids


def build_vocab(img_ids, min_words_threshold):
    counter = Counter()
    for id in tqdm(img_ids, 'Building vocabulary'):
        captions, _ = Flickr30kDataset.img_sent_ptype[id]
        for caption in captions:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            if tokens[-1] == '.':
                tokens = tokens[:-1]
            counter.update(tokens[:-1])

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= min_words_threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def build_flickr30k_job(job_tasks: Dict[str, List['str']], root_dir, min_words_threshold=5,
                        max_samples=Dict[str, int], shuffle=False, job_name=None):
    tasks_splits_img_ids = split_tasks_img_ids(job_tasks, root_dir, max_samples, shuffle)
    tasks = []
    for tname, task_splits_img_ids in tasks_splits_img_ids.items():
        tvocab = build_vocab(task_splits_img_ids['train'], min_words_threshold)
        T = Task(tname, tvocab, task_splits_img_ids['train'], task_splits_img_ids['val'], task_splits_img_ids['test'])
        tasks.append(T)
    return Job(tasks, name=job_name)


class Flickr30kJobFactory(JobFactory):
    def __init__(self, flickr30k_dir, job_dict, max_samples, job_name=None):
        self.max_samples = max_samples
        self._flickr30k_dir = flickr30k_dir
        self._job_dict = job_dict
        self._job_name = job_name if job_name is not None else '-'.join(self.job_dict.keys())
        self._job_name = 'flickr30k-' + job_name

    @property
    def name(self) -> str:
        return self._job_name

    @property
    def job_dict(self) -> dict:
        return self._job_dict

    def _build_job(self) -> Job:
        return build_flickr30k_job(self.job_dict, self._flickr30k_dir, min_words_threshold=5,
                                   max_samples=self.max_samples, job_name=self.name)
