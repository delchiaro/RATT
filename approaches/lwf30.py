from CNIC.utils import NP
from copy import deepcopy
from dataclasses import dataclass
from time import time

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence

from CNIC.approach import Approach
from CNIC.epoch_data import EpochData
from CNIC.model import EncoderCNN, DecoderLSTMCell
from CNIC.task import Job
from CNIC.utils import accuracy, freeze_all, AverageMeter

def pack_like(sequence_tensor: torch.Tensor, packed_sequence: PackedSequence):
    return PackedSequence(sequence_tensor, packed_sequence.batch_sizes, packed_sequence.sorted_indices,
                          packed_sequence.unsorted_indices)

def distillation_cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs, dim=1)
    tar = torch.nn.functional.softmax(targets, dim=1)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


class ApproachLwF_3(Approach):
    @dataclass
    class Settings(Approach.Settings):
        lwf_lambda: float = 1.0
        lwf_T: float = 1.0
        lwf_h_distillation: bool = False
        lwf_h_lambda: float = 1.0

    def __init__(self, encoder_cnn: EncoderCNN, decoder: DecoderLSTMCell,
                 settings: Settings = Settings(), device=None):
        settings.freeze_old_words=True
        super().__init__(encoder_cnn, decoder, settings, device)
        self.old_encoder_cnn = None
        self.old_decoder_rnn = None
        #self.old_classifiers = {}
        self.settings = settings

    def train_task(self, t: int, job: Job, start_ep=1, log_step=20, prev_task_epoch=None, monitor_metric='BLEU-4',
                   model_dir='models/', save_all_epochs=True, eval_all_epochs=True):
        if t > 0:
            # self.eval_job_bleu(job)
            self.old_encoder_cnn = deepcopy(self.encoder_cnn)
            self.old_decoder_rnn = deepcopy(self.decoder_rnn)
            self.old_decoder_rnn.eval()
            #self.old_classifiers[t] = self.old_decoder_rnn.linear
            freeze_all(self.old_decoder_rnn)

        return super().train_task(t, job, start_ep, log_step, prev_task_epoch, monitor_metric, model_dir,
                                  save_all_epochs, eval_all_epochs)

    def _encode_old(self, images_tensor):
        if len(images_tensor.shape) == 4:
            # if it's an image run cnn forward before shallow over features
            images_tensor = self.old_encoder_cnn(images_tensor)
        return images_tensor

    def _train_epoch(self, job, t, epoch, train_loader, active_words, log_step):
        if t > 0:
            # remapped_vocabs = [job.get_vocab_full_remapped(i) for i in range(0, len(job.tasks))]
            self.__new_task_full_map = torch.tensor(job.get_vocab_full_remapped(t), dtype=torch.long).to(self.device)

            old_words = set([w for task in job.tasks[:t] for w in task.vocab.words])
            self.__common_words = set(job.tasks[t].vocab.words).intersection(old_words)
            self.__common_words_ids = torch.tensor([job.full_vocab.wordmap[w] for w in self.__common_words])
            self.__common_words_ids_truthmap = torch.zeros([len(job.full_vocab)]).to(self.device)
            self.__common_words_ids_truthmap[self.__common_words_ids] = 1

            self.old_encoder_cnn.eval()
            self.old_decoder_rnn.eval()

        return super()._train_epoch(job, t, epoch, train_loader, active_words, log_step)

    def _new_epoch_data(self, train_loader, epoch):
        return EpochData(train_loader, epoch).add_loss('cce').add_loss('lwf').add_timer('lwf').add_loss('lwf_h')

    def _train_step(self, job, t, epoch, i, train_loader, images, targets, lengths, extra, active_words, epdata):
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)
        nb_words = lengths.sum()

        # Forward
        time_forward = time()
        features = self._encode(images)
        logits, (h, c), (h_states, c_states) = self.decoder_rnn.forward(features, targets, lengths, active_words)

        # Loss
        cce_loss = self.criterion(logits[:, active_words], packed_targets.data)
        epdata.upd_loss('cce', cce_loss.detach().cpu(), nb_words)
        epdata.upd_timer('forward', time_forward)

        # LWF
        if t > 0:
            time_lwf = time()
            full_targets = self.__new_task_full_map[targets]  # map targets to full targets

            self.old_decoder_rnn.word_embed = self.decoder_rnn.word_embed # using current word-embedding
            logits_old, (h, c), (h_states_old, c_states_olds) = self.old_decoder_rnn.forward(features, targets, lengths, active_words)
            logits_old = logits_old[:, self.old_words]
            logits_new = logits[:, self.old_words]

            #distil_loss = torch.nn.functional.l1_loss(logits_new, logits_old)
            distil_loss = distillation_cross_entropy(logits_new, logits_old, exp=1/self.settings.lwf_T)
            epdata.upd_loss('lwf', distil_loss.detach().cpu(), len(logits_new))

            loss = cce_loss  +  self.settings.lwf_lambda * distil_loss
            if self.settings.lwf_h_distillation:
                h_dist_loss = torch.nn.functional.l1_loss(h_states, h_states_old)
                loss = loss + self.settings.lwf_h_lambda * h_dist_loss
                epdata.upd_loss('lwf_h', h_dist_loss.detach().cpu(), len(h_states_old))

            epdata.upd_timer('lwf', time_lwf)

        else:
            loss = cce_loss

        epdata.upd_loss('loss', loss.detach().cpu(), nb_words)

        time_metric = time()
        # Metrics etc..
        epdata.upd_metric('top5', accuracy(logits[:, active_words], packed_targets.data, 5), nb_words)
        epdata.upd_metric('top1', accuracy(logits[:, active_words], packed_targets.data, 1), nb_words)
        packed_preds = PackedSequence(logits[:, active_words].argmax(-1), packed_targets.batch_sizes,
                                      packed_targets.sorted_indices, packed_targets.unsorted_indices)
        epdata.store_pred_caps(*pad_packed_sequence(packed_preds, batch_first=True))
        epdata.store_targets(*pad_packed_sequence(packed_targets, batch_first=True))
        epdata.upd_timer('metric', time_metric)

        time_backward = time()
        # Backward and Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.settings.freeze_old_words and t>0 and self.old_words is not None:
            self.decoder_rnn.word_embed.weight.grad[self.old_words] = 0

        self.optimizer.step()
        epdata.upd_timer('backward', time_backward)
