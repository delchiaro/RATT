from abc import ABC, abstractmethod
from collections import Counter
from os.path import join
from typing import Dict, List
import nltk
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

from CNIC.task import Task, Job, JobFactory
from CNIC.vocab import Vocabulary


def coco_tasks_img_ids(tasks: Dict[str, List['str']], coco_instances_annotation_path):
    # Loading COCO dataset with object detection labels:
    coco = COCO(coco_instances_annotation_path)
    img_ids = {}
    for task_name, task_cat_names in tasks.items():
        task_cat_ids = coco.getCatIds(task_cat_names)

        # For each category id in current task, get all images having at least an annotation with that category
        task_img_ids = [coco.getImgIds(catIds=[cat]) for cat in task_cat_ids]

        # flatten and remove duplicate:
        task_img_ids = set([t for tl in task_img_ids for t in tl])

        img_ids[task_name] = task_img_ids

    # remove from each task all the image that are also in another task
    commons = []
    for task_name, task_img_ids in img_ids.items():
        others = [k for k in img_ids.keys() if k != task_name]
        for other_task in others:
            commons += img_ids[other_task].intersection(task_img_ids)
    for task_name in img_ids.keys():
        img_ids[task_name] -= set(commons)
    return img_ids


def coco_split_tasks_img_ids(job_tasks: Dict[str, List['str']], coco_path, test_ratio=0.5, shuffle=False,
                             return_separate_split_tasks=False):
    print("\nComputing not intersecting image ids for current job (Training Set)")
    train_tasks_img_ids = coco_tasks_img_ids(job_tasks, join(coco_path, 'annotations/instances_train2014.json'))
    print("\nComputing not intersecting image ids for current job (Validation Set)")
    val_tasks_img_ids = coco_tasks_img_ids(job_tasks, join(coco_path, 'annotations/instances_val2014.json'))

    test_tasks_img_ids = None
    if test_ratio > 0 and test_ratio < 1:
        test_tasks_img_ids = {}
        print('\nSplitting validation set in test-val')
        for task, img_ids in val_tasks_img_ids.items():
            nb_test_imgs = int(len(img_ids) * test_ratio)
            img_ids = np.random.permutation(img_ids) if shuffle else img_ids
            test_tasks_img_ids[task] = set(list(img_ids)[:nb_test_imgs])
            val_tasks_img_ids[task] = set(list(img_ids)[nb_test_imgs:])

    if return_separate_split_tasks:
        return train_tasks_img_ids, val_tasks_img_ids, test_tasks_img_ids

    else:
        tasks_splits_img_ids = {}
        for task in train_tasks_img_ids.keys():
            tasks_splits_img_ids[task] = {}
            tasks_splits_img_ids[task]['train'] = train_tasks_img_ids[task]
            tasks_splits_img_ids[task]['val'] = val_tasks_img_ids[task]
            tasks_splits_img_ids[task]['test'] = test_tasks_img_ids[task] if test_tasks_img_ids is not None else None
        return tasks_splits_img_ids


def build_coco_vocab(coco_captions_annotation_path, img_ids, min_words_threshold=5):
    # Loading COCO dataset with captioning labels:
    coco = COCO(coco_captions_annotation_path)
    counter = Counter()
    ids = coco.getAnnIds(imgIds=img_ids)
    for i, id in enumerate(tqdm(ids, 'Building vocabulary')):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= min_words_threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def build_coco_job(job_tasks: Dict[str, List['str']], coco_path, test_ratio=0.5, test_shuffle=False,
                   min_words_threshold=5,
                   job_name=None):
    tasks_splits_img_ids = coco_split_tasks_img_ids(job_tasks, coco_path, test_ratio, test_shuffle)
    tasks = []
    for tname, task_splits_img_ids in tasks_splits_img_ids.items():
        tvocab = build_coco_vocab(join(coco_path, 'annotations/captions_train2014.json'), task_splits_img_ids['train'],
                                  min_words_threshold)
        T = Task(tname, tvocab, task_splits_img_ids['train'], task_splits_img_ids['val'], task_splits_img_ids['test'])
        tasks.append(T)
    return Job(tasks, name=job_name)


class CocoJobFactory(JobFactory):
    def __init__(self, coco_dir, job_dict, job_name=None):
        self._coco_dir = coco_dir
        self._job_dict = job_dict
        self._name = job_name if job_name is not None else '-'.join(self.job_dict.keys())
        self._name = 'coco-' + self._name

    @property
    def name(self) -> str:
        return self._name

    @property
    def job_dict(self) -> dict:
        return self._job_dict

    def _build_job(self) -> Job:
        return build_coco_job(self.job_dict, self._coco_dir, test_ratio=.5, test_shuffle=False,
                              min_words_threshold=5, job_name=self.name)
