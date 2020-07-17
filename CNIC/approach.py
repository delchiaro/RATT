from dataclasses import dataclass
import os
from os.path import join
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from CNIC.epoch_data import EpochData
from CNIC.task import Job, Task
from datasets.coco.coco_dataset import CocoDataset
from CNIC.model import DecoderLSTMCell, EncoderCNN
from CNIC.utils import accuracy, AverageMeter, VPrinter, CsvResults, create_file_dirs

from time import time
from dacite import from_dict




class Approach:

    @dataclass
    class Settings:
        args: dict = None
        lr: float = 1e-3
        wd: float = 0.0
        nb_epochs: int = 20
        first_task_extra_epochs: int = 0
        freeze_old_words: bool = False
        nlg_eval: bool = True

        @classmethod
        def from_dict(cls, data_dict):
            return from_dict(cls, data_dict)

        def to_dict(self):
            return self.__dict__

    def __init__(self, encoder_cnn: EncoderCNN, decoder: DecoderLSTMCell,
                 settings: Settings=Settings(), device=None):
        self._job = None
        self.encoder_cnn = encoder_cnn.to(device)
        self.decoder_rnn = decoder.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.settings = settings
        self.device = device
        self.optimizer = None
        self.init_optimizer()
        self.timer_verbose = False
        self.old_words = None
        self.nlgeval = None
        if self.settings.nlg_eval:
            from nlgeval import NLGEval
            print('loading nlg-eval... ', end='')
            self.nlgeval = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True)  # loads the models
            print('Done!')

    def init_optimizer(self):
        params = list(self.decoder_rnn.parameters())  # don't train CNN
        self.optimizer = torch.optim.Adam(params, lr=self.settings.lr, weight_decay=self.settings.wd)

    def train_job(self, job: Job, start_t=0, start_ep=1, log_step=20, monitor_metric='BLEU-4', model_dir='models/',
                  save_all_epochs=True, eval_all_epochs=True):
        print("Start training job with following settings:")
        print(self.settings, end='\n\n')
        start_ep = 1 if start_ep < 1 else start_ep
        start_t = 0 if start_t < 0 else start_t

        # By default doesn't load any checkpoint and starts from scratch
        load_t = None
        load_ep = None
        best_epoch = None

        # resume task training
        if start_ep > 1:
            load_t = start_t
            load_ep = start_ep-1

        # Start from first epoch but not from first task, i.e. load previous task and train on a new task
        elif start_t > 0:
            load_t = start_t - 1
            load_ep = 'best'
            start_ep = 1

        if load_t is not None:
            states = self.load_model_state(model_dir, t=load_t, epoch=load_ep, verbose=True)
            best_epoch = states['epoch'] if load_t == start_t-1 else None

        for t in list(range(len(job.tasks)))[start_t:]:
            print("\n\n" + "="*80 + f"\n TRAIN ON TASK {job.tasks[t].name}\n")
            _, _, best_epoch = self.train_task(t, job, start_ep, log_step, best_epoch, monitor_metric, model_dir,
                                               save_all_epochs, eval_all_epochs)
            print("\n" + "=" * 80 + f"\n TASK {job.tasks[t].name} COMPLETED.\n" + "="*80)
            CsvResults.merge_csv_results(model_dir, monitor_metric, metrics=['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr'], last_t=t)
            CsvResults.merge_csv_results(model_dir, monitor_metric, last_t=t)
            start_ep = 1  # next task will start from scratch

            # Load current task best model before starting next task
            self.load_model_state(model_dir, t=t, epoch='best', load_optim=False, load_cnn=False)


    def _get_active_words(self, job, t):
        active_words = job.full_vocab.remap_vocab_word_ids(job.tasks[t].vocab)
        return active_words

    def train_task(self, t: int, job: Job, start_ep=1, log_step=20, prev_task_epoch=None, monitor_metric='BLEU-4',
                   model_dir='models/', save_all_epochs=True, eval_all_epochs=True):
        task = job.tasks[t]
        train_loader = task.train_loader
        csv_val = CsvResults(self._csv_path(model_dir, t), 'VAL', [task.name for task in job.tasks])
        csv_val.init_csv(overwrite=True)

        if start_ep == 1: # reset optimizer if training a task from scratch
            self.init_optimizer()

        if t > 0: # Save old-word indices to freeze old word embeddings
            self.old_words = torch.cat([torch.tensor(self._get_active_words(job, i)) for i in range(0, t)])

        best_monitored_score = -1
        best_epdata, best_scores, best_epoch = None, None, 0

        active_words = self._get_active_words(job, t)
        nb_epochs = self.settings.nb_epochs
        if t == 0:
            nb_epochs += self.settings.first_task_extra_epochs

        for epoch in range(start_ep, nb_epochs + 1):

            epdata = self._train_epoch(job, t, epoch, train_loader, active_words, log_step)
            if save_all_epochs:
                self.save_model_state(model_dir, t, epoch)

            # Evaluate on validation sets of all tasks
            scores, _ = self.eval_job_metrics(job, sampling=True, verbose=True, tasks=None if eval_all_epochs else [t])

            # Check and write results
            csv_val.write_csv_row(scores, t, epoch, prev_task_epoch=prev_task_epoch, task_agnostic=True)

            obs_scores = scores[t] if eval_all_epochs else scores[0]
            if obs_scores[monitor_metric] > best_monitored_score:
                best_epoch = epoch
                best_epdata = epdata
                best_scores = scores[t]
                best_monitored_score = scores[t][monitor_metric]
                self.save_as_best_model(model_dir, t, epoch)

        return best_epdata, best_scores, best_epoch

    def _encode(self, image_tensor):
        """
        If it's an image (batch channel + rgb channels): run cnn forward and return features.
        If it's not an image, then it's already a pre-extracted feature vector: return it.
        """
        if len(image_tensor.shape) == 4:
            image_tensor = self.encoder_cnn(image_tensor)
        return image_tensor

    def _new_epoch_data(self, train_loader, epoch) -> EpochData:
        """
        Prepare the EpocData object for current approach.
        Subclasses could override this method and return an EpochData object created with different parameters.
        """
        return EpochData(train_loader, epoch)

    def _train_epoch(self, job, t, epoch, train_loader, active_words, log_step):
        self.encoder_cnn.train()
        self.decoder_rnn.train()
        epdata = self._new_epoch_data(train_loader, epoch)
        time_batch = time()
        for i, (images, targets, lengths, extra) in enumerate(train_loader):
            # Set mini-batch dataset
            images = images.to(self.device)
            targets = targets.to(self.device)
            epdata.upd_timer('data', time_batch)

            self._train_step(job, t, epoch, i, train_loader, images, targets, lengths, extra, active_words, epdata)

            # Print log info
            if (i+1) % log_step == 0:
                epdata.print_batch_status(i+1, all_timers=self.timer_verbose)

            epdata.upd_timer('batch', time_batch)
            time_batch = time()
        return epdata

    def _train_step(self, job, t, epoch, i, train_loader, images, targets, lengths, extra, active_words, epdata):
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)
        nb_words = lengths.sum()

        # Forward
        time_forward = time()
        features = self._encode(images)
        logits, _, _ = self.decoder_rnn.forward(features, targets, lengths, active_words)

        # Loss
        loss = self.criterion(logits[:, active_words], packed_targets.data)
        epdata.upd_loss('loss', loss.detach().cpu(), nb_words)
        epdata.upd_timer('forward', time_forward)

        # Metrics etc..
        time_metric = time()
        epdata.upd_metric('top5', accuracy(logits[:, active_words], packed_targets.data, 5), nb_words)
        epdata.upd_metric('top1', accuracy(logits[:, active_words], packed_targets.data, 1), nb_words)
        packed_preds = PackedSequence(logits[:, active_words].argmax(-1), packed_targets.batch_sizes,
                                      packed_targets.sorted_indices, packed_targets.unsorted_indices)
        epdata.store_pred_caps(*pad_packed_sequence(packed_preds, batch_first=True))
        epdata.store_targets(*pad_packed_sequence(packed_targets, batch_first=True))
        epdata.upd_timer('metric', time_metric)

        # Backward and Optimize
        time_backward = time()
        self.optimizer.zero_grad()
        loss.backward()
        if self.settings.freeze_old_words and t>0 and self.old_words is not None:
            self.decoder_rnn.word_embed.weight.grad[self.old_words] = 0
        self.optimizer.step()
        epdata.upd_timer('backward', time_backward)

    def eval_job(self, job: Job, sampling=True, verbose=True, tasks=None, aware_t=None):
        vprint = VPrinter(verbose)
        job_val_data = []
        tasks = [job.tasks[t] for t in tasks] if tasks is not None else job.tasks
        for val_t, val_task in enumerate(tasks):
            t = aware_t if aware_t is not None else val_t
            vprint(f"Evaluate task {val_t} ({val_task.name}) with task awareness {t}")
            epdata = self.evaluate(t, job, val_task.val_loader, sampling)
            job_val_data.append(epdata)
        return job_val_data

    def eval_job_metrics(self, job: Job, bleu=None, sampling=True, verbose=True, tasks=None, aware_t=None):
        vprint = VPrinter(verbose)
        job_scores = []
        job_val_data = []
        tasks = [job.tasks[t] for t in tasks] if tasks is not None else job.tasks
        for val_t, val_task in enumerate(tasks):
            vprint(f"Evaluate task {val_task.name}")
            metrics, val_data = self.eval_metrics(val_t, job, val_task.val_loader, bleu, sampling, aware_t=aware_t)
            vprint(f"BLEU-1   BLEU-2   BLEU-3   BLEU-4   METEOR   ROUGE_L   CIDEr   TOP-1   TOP-5")
            vprint(f"{metrics['BLEU-1']:.4f}", end='   ')
            vprint(f"{metrics['BLEU-2']:.4f}", end='   ')
            vprint(f"{metrics['BLEU-3']:.4f}", end='   ')
            vprint(f"{metrics['BLEU-4']:.4f}", end='   ')
            vprint(f"{metrics['METEOR']:.4f}", end='   ')
            vprint(f"{metrics['ROUGE_L']:.4f}", end='    ')
            vprint(f"{metrics['CIDEr']:.4f}", end='  ')
            vprint(f"{metrics['TOP-1']:.4f}", end='  ')
            vprint(f"{metrics['TOP-5']:.4f}")
            vprint("")

            job_val_data.append(val_data)
            job_scores.append(metrics)
        return job_scores, job_val_data


    def evaluate(self, t, job: Job, data_loader: DataLoader, sampling=True):
        self.encoder_cnn.eval()
        self.decoder_rnn.eval()
        task = job.tasks[t]
        active_word_ids = job.full_vocab.remap_vocab_word_ids(task.vocab)
        epdata = EpochData(data_loader)
        for i, (images, targets, lengths, extra) in enumerate(tqdm(data_loader)):
            # Set mini-batch dataset
            images = images.to(self.device)
            targets = targets.to(self.device)

            # When data loader return all_targets per each image,
            # we will have to repeat the predicted logits on batch size axis to match the number of targets
            repeat = None
            if len(targets.shape) == 3:
                repeat = targets.shape[1]
                targets = targets.view(-1, targets.shape[-1])
                lengths = lengths.view(-1)

            packed_targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False, )

            # Forward
            features = self._encode(images)
            if sampling:
                decoded_ids, logits, decode_lens = self.decoder_rnn.sample(features, map_words=active_word_ids)
                epdata.store_pred_caps(decoded_ids, decode_lens)

                # For loss computation we compute pckd_logits, eventually extending logits in time-dimension
                logits = logits.repeat_interleave(repeat, 0) if repeat is not None else logits
                if targets.shape[1] > logits.shape[1]:
                    l = torch.zeros(logits.shape[0], targets.shape[1], logits.shape[2]).to(self.device)
                    l[:, :logits.shape[1], :] = logits
                    logits = l
                pckd_logits = pack_padded_sequence(logits, lengths, batch_first=True, enforce_sorted=False).data

            else:
                features = features.repeat_interleave(repeat, 0) if repeat is not None else features
                pckd_logits = self.decoder_rnn.forward(features, targets, lengths, active_word_ids, enforce_sorted=False)
                packed_preds = PackedSequence(pckd_logits[:, active_word_ids].argmax(-1),
                                              packed_targets.batch_sizes, packed_targets.sorted_indices,
                                              packed_targets.unsorted_indices)
                decoded_ids, decode_lens = pad_packed_sequence(packed_preds, batch_first=True)
                epdata.store_pred_caps(decoded_ids, decode_lens)

            # Loss
            nb_words = lengths.sum()

            loss = self.criterion(pckd_logits, packed_targets.data)
            epdata.upd_loss('loss', loss.detach().cpu(), nb_words)

            # Accuracy
            epdata.upd_metric('top5', accuracy(pckd_logits[:, active_word_ids], packed_targets.data, 5), nb_words)
            epdata.upd_metric('top1', accuracy(pckd_logits[:, active_word_ids], packed_targets.data, 1), nb_words)
            epdata.store_targets(targets, lengths)
            epdata.extra_data.append(extra)
        return epdata

    def eval_metrics(self, t, job: Job, data_loader, bleu_idx=None, sampling=True, aware_t=None):
        from CNIC.utils import bleu_score
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

        bleu_idx = (1, 2, 3, 4) if bleu_idx is None else bleu_idx
        bleu_idx = [bleu_idx] if isinstance(bleu_idx, int) else bleu_idx
        preds_t = aware_t if aware_t is not None else t
        epdata = self.evaluate(preds_t, job, data_loader, sampling)
        preds = epdata.get_pred_caps()
        targets = epdata.get_target_caps()
        if self.nlgeval is not None:
            translated_preds = job.tasks[preds_t].vocab.translate_all(preds)
            translated_targets = [job.tasks[t].vocab.translate_all(trgts) for trgts in targets]
            refs = [ [trg[i] for trg in translated_targets] for i in range(len(translated_targets[0]))]
            metrics = self.nlgeval.compute_metrics(refs, translated_preds)
            metrics = {'BLEU-1': metrics['Bleu_1'], 'BLEU-2': metrics['Bleu_2'],
                       'BLEU-3': metrics['Bleu_3'], 'BLEU-4': metrics['Bleu_4'],
                       'METEOR': metrics['METEOR'], 'ROUGE_L': metrics['ROUGE_L'],
                       'CIDEr': metrics['CIDEr'],
                       'TOP-1': float(epdata.metric('top1').avg/100),
                       'TOP-5': float(epdata.metric('top5').avg/100)}

        else:
            translated_preds = job.tasks[preds_t].vocab.translate_all(preds)
            translated_targets = [job.tasks[t].vocab.translate_all(trgts) for trgts in targets]
            bleu = {b: bleu_score(targets, preds, bleu=b) for b in bleu_idx}

            meteor = np.mean([meteor_score(trg, prd) for trg, prd in zip(translated_targets, translated_preds)])

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)
            rouge_l_all = np.array([np.array([scorer.score(tr, prd)['rougeL'][:] for tr in trg])
                               for trg, prd, in zip(translated_targets, translated_preds)])
            rouge_precision, rouge_recall, rouge_fscore = rouge_l_all.max(axis=1).mean(axis=0)

            metrics = {'BLEU-1': bleu[1], 'BLEU-2': bleu[2], 'BLEU-3': bleu[3], 'BLEU-4': bleu[4],
                       'METEOR': meteor, 'ROUGE_L':rouge_recall,
                       'CIDEr': -1, # Not computed
                       'TOP-1': float(epdata.metric('top1').avg/100),
                       'TOP-5': float(epdata.metric('top1').avg/100)}

        return metrics, epdata



    @staticmethod
    def _model_state_path(model_dir, t, epoch, batch=None):
        epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else epoch
        fname = f'_task-{t:02d}_epoch-{epoch_str}'
        fname += f'_b-{batch:05d}' if batch is not None else ''
        return join(model_dir, f'model{fname}.ckpt')

    def _csv_path(self, model_dir, t):
        return join(model_dir, f"results_t-{t:02d}.csv")

    def save_as_best_model(self, model_dir, t, epoch_number):
        self.save_model_state(model_dir, t, epoch='best')
        self._save_best_epoch_info(model_dir, t, epoch_number, epoch_name='best')

    def _save_best_epoch_info(self, model_dir, t, epoch_number, epoch_name='best'):
        path = model_dir if t is None else self._model_state_path(model_dir, t, epoch_name)
        with open(path + '.info', 'w') as f:
            f.write(f"epoch={epoch_number}\n")
            f.write(f"settings={self.settings.to_dict()}\n")

    def _get_save_state(self, job, t, epoch, batch, save_optim, save_cnn):
        state = {'decoder_rnn': self.decoder_rnn.state_dict(), 't': t, 'epoch': epoch, 'batch': batch}
        state['settings'] = self.settings.to_dict()
        if save_optim: state['optim'] = self.optimizer.state_dict()
        if save_cnn: state['encoder_cnn'] = self.encoder_cnn.state_dict()
        return state

    def save_model_state(self, model_dir, t, epoch, batch=None, save_optim=True, save_cnn=False):
        path = model_dir if t is None else self._model_state_path(model_dir, t, epoch, batch)
        create_file_dirs(path)
        state = self._get_save_state(self._job, t, epoch, batch, save_optim, save_cnn)
        torch.save(state, path)

    def load_model_state(self, model_dir, t=None, epoch=None, batch=None,
                         load_optim=None, load_cnn=None, load_settings=False, verbose=False):
        vprint = VPrinter(verbose)
        path = model_dir if t is None else self._model_state_path(model_dir, t, epoch, batch)
        vprint(f'Loading model {path}')
        states = torch.load(path, map_location=self.device)
        self.decoder_rnn.load_state_dict(states['decoder_rnn'])
        load_cnn = load_cnn if load_cnn is not None else 'encoder_cnn' in states.keys()
        load_optim = load_optim if load_optim is not None else 'optim' in states.keys()
        if load_settings: self.settings = from_dict(self.settings.__class__, states['settings'])
        if load_cnn: self.encoder_cnn.load_state_dict(states['encoder_cnn'])
        if load_optim: self.optimizer.load_state_dict(states['optim'])
        return states
