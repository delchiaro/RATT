from tqdm import tqdm

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



class ApproachEWC(Approach):

    @dataclass
    class Settings(Approach.Settings):
        ewc_sampling_type: str = 'multinomial'  #'true' 'max_pred' 'multinomial'
        ewc_teacher_forcing: bool = False
        ewc_lambda: float = 5.0


    def __init__(self, encoder_cnn: EncoderCNN, decoder: DecoderLSTMCell,
                 settings: Settings = Settings(), device=None):
        super().__init__(encoder_cnn, decoder, settings, device)
        self.settings = settings
        self.fisher = None
        self.old_params = None


    def _get_save_state(self, job, t, epoch, batch, save_optim, save_cnn):
        state = super()._get_save_state(job, t, epoch, batch, save_optim, save_cnn)
        state['fisher'] = self.fisher
        state['older_params'] = self.old_params
        return state


    def train_task(self, t: int, job: Job, start_ep=1, log_step=20, prev_task_epoch=None, monitor_metric='BLEU-4',
                   model_dir='models/', save_all_epochs=True, eval_all_epochs=True):
        if t > 0:
            self.old_words = torch.cat([torch.tensor(self._get_active_words(job, i)) for i in range(0, t)])
            active_words = self._get_active_words(job, t)
            train_loader = job.tasks[t].train_loader
            self.older_params = self.get_older_params()
            fisher = self.fisher_matrix_diag(train_loader, active_words)
            if t == 1:
                self.fisher = fisher
            else:
                for n in self.fisher.keys():
                    self.fisher[n] = self.fisher[n] + fisher[n]

        return super().train_task(t, job, start_ep, log_step, prev_task_epoch, monitor_metric, model_dir,
                                  save_all_epochs, eval_all_epochs)

    def get_older_params(self):
        return {n: p.clone().detach() for n, p in self.get_ewc_params()}

    def get_ewc_params(self):
        params = {name: param for name, param in self.decoder_rnn.named_parameters() if param.requires_grad}

        return params.items()

    def fisher_matrix_diag(self, train_loader, active_words):
        self.encoder_cnn.eval()
        self.decoder_rnn.eval()
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self.get_ewc_params() if p.requires_grad}

        not_old_words = sorted(set(range(self.decoder_rnn.word_embed.weight.shape[0])) - set(self.old_words.numpy().tolist()))
        not_old_words = torch.tensor(not_old_words, dtype=torch.long)
        nb_words = 0
        time_batch = time()
        for i, (images, targets, lengths, extra) in enumerate(tqdm(train_loader, "Computing EWC fisher matrix")):
            # Set mini-batch dataset
            targets = targets.to(self.device)
            images = images.to(self.device)
            bs = len(images)
            nb_words += lengths.sum()

            packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)
            features = self._encode(images)

            if self.settings.ewc_teacher_forcing:
                logits, _, _ = self.decoder_rnn.forward(features, targets, lengths, active_words)
                logits = logits[:, active_words] # remove out-of-vocab logits
            else:
                #_, logits, _ = self.decoder_rnn.sample(features, max_seq_len=max(lengths))
                _, logits, _ = self.decoder_rnn.sample(features, map_words=active_words, max_seq_len=max(lengths))
                #logits = pack_like(logits, packed_targets)
                logits = logits[:, :, active_words]  # remove out-of-vocab logits
                logits = pack_padded_sequence(logits, lengths, batch_first=True).data



            # Forward
            if self.settings.ewc_sampling_type == 'true':
                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = packed_targets.data

            elif self.settings.ewc_sampling_type == 'max_pred':
                # Not use labels and compute the gradients related to the prediction the model has learned
                preds = pack_like(logits.argmax(-1), packed_targets).data

            elif self.settings.ewc_sampling_type == 'multinomial':
                # Use a multinomial sampling to compute the gradients
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.multinomial(probs, 1).flatten()
                preds = pack_like(preds, packed_targets).data
            else:
                raise ValueError(f"Sampling type value '{self.settings.ewc_sampling_type}' is not valid.")

            # Loss
            loss = self.criterion(logits, preds)
            loss.backward()

            # put to zero the gradient for all the new weights
            # (we don't wont to regularize weights related to new words!)
            self.decoder_rnn.word_embed.weight.grad[not_old_words] = 0.
            self.decoder_rnn.linear.weight.grad[not_old_words] = 0.
            self.decoder_rnn.linear.bias.grad[not_old_words] = 0.

            for n, p in self.get_ewc_params():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * bs * lengths.sum()

        fisher = {n: (p / nb_words) for n, p in fisher.items()}
        #fisher = {n: (p / len(train_loader)) for n, p in fisher.items()}
        return fisher

    def _new_epoch_data(self, train_loader, epoch):
        return EpochData(train_loader, epoch).add_loss('ewc').add_loss('cce')

    def _train_step(self, job, t, epoch, i, train_loader, images, targets, lengths, extra, active_words, epdata):
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)
        nb_words = lengths.sum()

        time_forward = time()

        # Forward
        features = self._encode(images)
        logits, _, _ = self.decoder_rnn.forward(features, targets, lengths, active_words)

        # Loss
        loss = cce = self.criterion(logits[:, active_words], packed_targets.data)
        if t > 0:
            ewc_reg = self.regularizer() * self.settings.ewc_lambda
            epdata.upd_loss('ewc', ewc_reg.detach().cpu())
            loss = cce + ewc_reg

        epdata.upd_loss('cce', cce.detach().cpu(), nb_words)
        epdata.upd_loss('loss', loss.detach().cpu(), nb_words)
        epdata.upd_timer('forward', time_forward)

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


    def regularizer(self):
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self.get_ewc_params():
            if n in self.fisher.keys():
                loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
        # print(loss_reg*self.lamb)
        return loss_reg