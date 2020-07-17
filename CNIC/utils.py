import csv
import os
import warnings
from os.path import join
from typing import List

import torch
import numpy as np
import random

from nltk.translate.bleu_score import corpus_bleu



def create_file_dirs(file_path):
    os.makedirs('/'.join(file_path.split('/')[:-1]), exist_ok=True)

def set_seed(seed):
    print(f"Using seed: {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_device(gpu=None):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def NP(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def freeze_all(module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False
    return

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)




def bleu_score(true_captions, pred_captions, bleu=4):
    assert bleu in [1, 2, 3, 4]
    weights = [1 / bleu] * bleu + [0] * (4 - bleu)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = corpus_bleu(true_captions, pred_captions, weights=weights)
    return scores



class AverageMeter(object):
    """ Keeps track of most recent, average, sum, and count of a metric. """

    def __init__(self, print_value=True, print_avg=True):
        self.print_value = print_value
        self.print_avg = print_avg
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def str(self, fp=3, mul=1, value=None, avg=None):
        value = self.print_value if value is None else value
        avg = self.print_avg if avg is None else avg
        value_str = f"{self.val*mul:.{fp}f}" if value else ''
        avg_str = f"({self.avg * mul:.{fp}f})" if avg else ''
        value_str = value_str + ' ' if avg and value else value_str
        return value_str + avg_str

    def __str__(self):
        return self.str()



class CsvResults:
    ALL_METRICS = ('BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR','ROUGE_L',   'CIDEr',   'TOP-1',   'TOP-5')
    FIELDS = ('split', 'all-tasks', 'training-task', 'prev-task-epoch', 'current-epoch', 'batch-idx', 'task-knowledge')
    def __init__(self, path, split: str, all_tasks: List[str],
                 metrics=ALL_METRICS):
        self.metrics = metrics
        self.all_tasks = all_tasks
        self.split = split
        self.csv_fields = self._get_fieldnames(self.all_tasks, self.metrics)
        self.fname = path
        create_file_dirs(path)

    @staticmethod
    def _get_fieldnames(all_tasks, metrics=ALL_METRICS):
        return list(CsvResults.FIELDS) + [f'{task}-{metric}' for task in all_tasks for metric in metrics]

    def init_csv(self, overwrite=True):
        if not os.path.isfile(self.fname) or overwrite:
            csv_file = open(self.fname, 'w')
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_fields)
            csv_writer.writeheader()
            csv_file.close()
        return self

    def write_csv_row(self, all_task_scores, t, epoch, prev_task_epoch='', batch='last', task_agnostic=False):
        training_task = self.all_tasks[t]
        csv_file = open(self.fname, 'a')
        csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_fields)
        row = {'split': self.split, 'all-tasks': self.all_tasks, 'training-task': training_task,
               'prev-task-epoch': prev_task_epoch, 'current-epoch': epoch, 'batch-idx': batch,
               'task-knowledge': 'agnostic' if task_agnostic else 'aware'}

        for val_t, scores in enumerate(all_task_scores):
            val_task = self.all_tasks[val_t]
            for metric, score in scores.items():
                row[f'{val_task}-{metric}'] = score

        csv_writer.writerow(row)
        csv_file.close()
        return self

    @staticmethod
    def merge_csv_results(out_path, monitor_metric='BLEU-4', metrics=ALL_METRICS, last_t=None):
        all_rows = []
        improving_rows = []
        best_rows = []
        files = os.listdir(out_path)
        csv_filenames = sorted([f for f in files if f.startswith('results_t-') and f.endswith('.csv')])
        last_t = np.inf if last_t is None else last_t
        all_tasks = None
        for csv_fname in csv_filenames:
            bestscore = 0
            best_row_idx = 0
            t = int(csv_fname.split('.csv')[0].split('results_t-')[1])
            if t <= last_t:
                with open(join(out_path, csv_fname), newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    fieldnames = reader.fieldnames
                    rows = []
                    for i, row in enumerate(reader):
                        if all_tasks is None:
                            all_tasks = [tname.split("'")[1] for tname in row['all-tasks'].split('[')[1].split(']')[0].split(',')]
                        task = all_tasks[t]
                        rows.append(row)
                        score = float(row[f"{task}-{monitor_metric}"])
                        if score > bestscore:
                            bestscore = score
                            best_row_idx = i
                    all_rows += rows
                    improving_rows += rows[:best_row_idx+1]
                    best_rows.append(rows[best_row_idx])
        append_fname = '' if len(metrics) == CsvResults.ALL_METRICS else '_' + ('_').join(metrics)

        fieldnames = CsvResults._get_fieldnames(all_tasks, metrics)
        all_rows = [{f: row[f] for f in fieldnames} for row in all_rows]
        improving_rows = [{f: row[f] for f in fieldnames} for row in improving_rows]
        best_rows = [{f: row[f] for f in fieldnames} for row in best_rows]

        with open(join(out_path, f'results_all{append_fname}.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        with open(join(out_path, f'results_best{append_fname}.csv'), 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(improving_rows)

            csvfile.write("\n\n")
            writer.writeheader()
            writer.writerows(best_rows)





class VPrinter:
    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self,  *args, sep=' ', end='\n', file=None):
        if self.verbose:
            print(*args, sep=sep, end=end, file=file)
