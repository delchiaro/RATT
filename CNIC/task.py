from abc import abstractmethod, ABC
from typing import List, Union, Dict, Optional

import pickle

from torch.utils.data import DataLoader

from CNIC.vocab import Vocabulary


class Task:

    def __init__(self, name: str, vocab: Vocabulary, train_ids: List[int],
                 val_ids: List[int] = None, test_ids: List[int] = None,
                 sort_ids=True):
        self.name = name
        self.vocab = vocab
        self.train_examples_ids = sorted(train_ids) if sort_ids and train_ids is not None else train_ids
        self.val_examples_ids = sorted(val_ids) if sort_ids and val_ids is not None else val_ids
        self.test_examples_ids = sorted(test_ids) if sort_ids and test_ids is not None else test_ids
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        # TODO: for now we only use val_loader, when testing on testset we load test_loader on val_loader variable.
        #self.test_loader: Optional[DataLoader] = None

    def __str__(self):
        return f"Task {self.name} [vocab-size: {len(self.vocab)}, train-imgs: {len(self.train_examples_ids)}, " \
               f"val-imgs: {len(self.val_examples_ids)}, test-imgs: {len(self.test_examples_ids)}]"


class Job:
    """
    A Job represent a list of tasks for a given dataset
    """
    def __init__(self, tasks: List[Task], name: str = None):
        self.name: str = name
        self.tasks: List[Task] = tasks
        self.tasks_dict: Dict[str, Task] = {task.name: task for task in tasks}
        self.full_vocab = Vocabulary()
        self.compute_full_vocab()
        self.__optimized_tasks_dict: Dict[Union[str, int], Task] =\
            {**self.tasks_dict, **{t: task for t, task in enumerate(self.tasks)}}

    def compute_full_vocab(self):
        self.full_vocab = Vocabulary()
        for task in self.tasks:
            for w in task.vocab.wordmap.keys():
                self.full_vocab.add_word(w)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def get_task(self, t: Union[str, int]) -> Task:
        return self.__optimized_tasks_dict[t]

    def get_vocab(self, t: Union[str, int]) -> Vocabulary:
        return self.__optimized_tasks_dict[t].vocab

    def get_vocab_full_remapped(self, t: Union[str, int]) -> Vocabulary:
        return self.full_vocab.remap_vocab_word_ids(self.__optimized_tasks_dict[t].vocab)
        # return self.__optimized_tasks_dict[t].vocab.remap_vocab_word_ids(self.full_vocab)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        '\n'.join([str(task) for task in self.tasks])



class JobFactory(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def fname(self) -> str:
        return self.name + '.job.pkl'

    @property
    @abstractmethod
    def job_dict(self) -> dict:
        pass

    @abstractmethod
    def _build_job(self) -> Job:
        pass

    def build_job(self, save=True) -> Job:
        job = self._build_job()
        if save:
            job.save(self.fname)
        return job

    def load_job(self) -> Job:
        return Job.load(self.fname)

    def get_job(self):
        try:
            job = self.load_job()
            print('Job loaded from disk.')
        except FileNotFoundError:
            print('Job not found. Building new job with vocabulary...')
            job = self.build_job(save=True)
            print('Job saved on disk.\n')

        print(f'Job {self.name} [full-vocab-size: {len(job.full_vocab)}]')
        for task, task_classes in zip(job.tasks, self.job_dict.values()):
            print(task)
            print(task_classes, end='\n\n')

        return job