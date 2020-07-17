from CNIC.utils import NP, VPrinter
from copy import deepcopy
from dataclasses import dataclass
from time import time

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
import numpy as np

from CNIC.approach import Approach
from CNIC.epoch_data import EpochData
from CNIC.model import EncoderCNN, DecoderLSTMCell
from CNIC.task import Job
from CNIC.utils import accuracy, freeze_all, AverageMeter
from CNIC.vocab import Vocabulary


class HATMask(nn.Module):
    def __init__(self, nb_tasks, size, default_s, usage=.5):
        super().__init__()
        self.default_s = default_s
        self.size = size
        self.task_embedding = nn.Embedding(nb_tasks, size)
        self.gate = nn.Sigmoid()
        self.reset_parameters(usage)

    def reset_parameters(self, usage=.5):
        # we add to random data a bias so that in average 'usage'*100% of the masks are > 0.5
        # To do so we compute the inverse of logistic function. For 50% usage (usage=0.5) bias is 0.
        bias = np.log(usage/(1-usage))
        self.task_embedding.reset_parameters()
        self.task_embedding.weight.data += bias  # we want the output mask to start from 1

    def forward(self, t, s=None, binarize=False):
        s = self.default_s if s is None else s
        h = self.task_embedding(t)
        mask = self.gate(s*h)
        if binarize:
            mask = (mask>.5).float()
        return mask


class DecoderLSTMCellRATT(DecoderLSTMCell):
    def __init__(self, visual_feat_size, embed_size, hidden_size, vocab: Vocabulary, nb_tasks, default_hat_s, max_decode_length=20, hat_usage=.5):
        """Set the hyper-parameters and build the layers."""
        super(DecoderLSTMCellRATT, self).__init__(visual_feat_size, embed_size, hidden_size, vocab, max_decode_length)
        self.hat_mask_emb = HATMask(nb_tasks, embed_size, default_hat_s, hat_usage)
        self.hat_mask_lstm = HATMask(nb_tasks, hidden_size, default_hat_s, hat_usage)
        # self.hat_mask_cls = HATMask(nb_tasks, len(vocab), default_hat_s)
        self.nb_tasks = nb_tasks
        self._default_hat_s = default_hat_s
        self._t = 0
        self._s = 1
        self._sample_binarize=False


    def get_hat_view_for(self, n, emb_mask, h_mask, c_mask):

        if n == 'visual_embed.weight':
            # Visual embedding is the first layer, it only has post mask.
            # We could have a pre-mask but we will have to apply a new HAT mask to ResNet output feature.
            post = emb_mask.view(-1, 1).expand_as(self.visual_embed.weight)
            return post
        elif n == 'visual_embed.bias':
            return emb_mask.data.view(-1)

        elif n == 'word_embed.weight':
            # Word Embedding previous layer is classifier
            post = emb_mask.view(1, -1).expand_as(self.word_embed.weight)
            pre = c_mask.data.view(-1, 1).expand_as(self.word_embed.weight)
            return torch.min(post, pre)

        elif n == 'lstm_cell.weight_hh':
            mask_lstm_h = h_mask.data.repeat([1, 4])  # repeat the mask per each gate (forget, input, output, g)
            post = mask_lstm_h.data.view(-1, 1).expand_as(self.lstm_cell.weight_hh)
            pre = h_mask.data.view(1, -1).expand_as(self.lstm_cell.weight_hh)
            return torch.min(post, pre)
            #return post

        elif n == 'lstm_cell.bias_hh':
            mask_lstm_h = h_mask.data.repeat([1, 4])  # repeat the mask per each gate (forget, input, output, g)
            return mask_lstm_h.data.view(-1, ).expand_as(self.lstm_cell.bias_hh)

        elif n == 'lstm_cell.weight_ih':
            mask_lstm_h = h_mask.data.repeat([1, 4])  # repeat the mask per each gate (forget, input, output, g)
            post = mask_lstm_h.data.view(-1, 1).expand_as(self.lstm_cell.weight_ih)
            pre = emb_mask.data.view(1, -1).expand_as(self.lstm_cell.weight_ih)
            return torch.min(post, pre)
        elif n == 'lstm_cell.bias_ih':
            mask_lstm_h = h_mask.data.repeat([1, 4])  # repeat the mask per each gate (forget, input, output, g)
            return mask_lstm_h.data.view(-1, ).expand_as(self.lstm_cell.bias_ih)

        elif n == 'linear.weight':
            post = c_mask.data.view(-1, 1).expand_as(self.linear.weight)
            pre =  h_mask.data.view(1, -1).expand_as(self.linear.weight)
            return torch.min(post, pre)
        elif n == 'linear.bias':
            return c_mask.data.view(-1)

        return None

    def init_ratt_forward(self, t, s, sample_binarize=False):
        self._t = torch.tensor(t, device=self.linear.weight.device)
        self._s = s
        self._sample_binarize = sample_binarize

    def forward(self, features, target_caps, lengths, map_words=None, enforce_sorted=True):
        """Decode image feature vectors and generates captions."""

        # HAT
        h_mask = self.hat_mask_lstm.forward(self._t, self._s)
        emb_mask = self.hat_mask_emb.forward(self._t, self._s)

        if map_words is not None:
            target_caps = torch.tensor(map_words, dtype=torch.long).to(features.device)[target_caps]
        vemb = self.visual_embed(features)
        wembs = self.word_embed(target_caps)
        embeddings = torch.cat((vemb.unsqueeze(1), wembs), 1)
        embeddings *= emb_mask.unsqueeze(0).repeat([embeddings.shape[0], embeddings.shape[1], 1])

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=enforce_sorted)
        h = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        c = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        h = h * h_mask  # HAT
        first=0
        h_states = []
        c_states = []
        for bs in packed.batch_sizes:
            x = packed.data[first:first+bs]
            h, c = self.lstm_cell.forward(x[:bs], (h[:bs], c[:bs]))
            h = h * h_mask  # HAT
            first = first+bs
            h_states.append(h)
            c_states.append(c)
        h_states = torch.cat(h_states, dim=0)
        c_states = torch.cat(c_states, dim=0)
        logits = self.linear(h_states)
        # packed_logits = PackedSequence(logits, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
        # h_states = PackedSequence(h_states, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
        # c_states = PackedSequence(c_states, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
        return logits, (h, c), (h_states, c_states), (emb_mask, h_mask)

    def sample(self, features, states=None, map_words=None,  max_seq_len=None):
        """Generate captions for given image features using greedy search."""

        s = self._default_hat_s
        # RATT
        h_mask = self.hat_mask_lstm.forward(self._t, s, self._sample_binarize)
        emb_mask = self.hat_mask_emb.forward(self._t, s, self._sample_binarize)

        if map_words is None:
            map_words = torch.arange(self.linear.weight.shape[0]).to(features.device)
        else:
            map_words = torch.tensor(map_words, dtype=torch.long).to(features.device)
        decoded_ids = []
        all_logits = []
        #inputs = features.unsqueeze(1)
        inputs = self.visual_embed(features)
        max_seq_len = self.max_seq_length if max_seq_len is None else max_seq_len
        decode_lens = torch.ones(features.shape[0], dtype=torch.long).to(features.device)*max_seq_len
        h = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        c = torch.zeros(len(features), self.lstm_cell.hidden_size, device=self.lstm_cell.weight_hh.device)
        h *= h_mask
        for i in range(max_seq_len):
            inputs *= emb_mask  #  RATT
            h, c = self.lstm_cell.forward(inputs, (h, c)) #  hiddens: (batch_size, hidden_size)
            h *= h_mask  # HAT
            logits = self.linear(h)  # logits:  (batch_size, vocab_size)
            predicted_selected = logits[:, map_words].argmax(1)  # predicted: (batch_size)
            decoded_ids.append(predicted_selected)
            predicted = map_words[predicted_selected]  # translate back to full word dict
            all_logits.append(logits)
            inputs = self.word_embed(predicted)  # inputs: (batch_size, embed_size)
            decode_lens[((predicted == Vocabulary.END_IDX) & (decode_lens == self.max_seq_length)).nonzero()] = i + 1

        return torch.stack(decoded_ids, 1), torch.stack(all_logits, 1), decode_lens




class ApproachRATT(Approach):
    @dataclass
    class Settings(Approach.Settings):
        lambd: float = 5000
        smax: float = 400
        thres_cosh: float = 50
        binary_backward_masks: bool = True
        binary_sample_forward_masks: bool = False
        usage: float = 50  # percentages (50.0 = 50%)

    def __init__(self, encoder_cnn: EncoderCNN, decoder: DecoderLSTMCellRATT,
                 settings: Settings = Settings(), device=None):
        assert isinstance(decoder, DecoderLSTMCellRATT)
        super().__init__(encoder_cnn, decoder, settings, device)
        self.decoder: DecoderLSTMCellRATT = decoder
        self.settings: ApproachRATT.Settings = settings
        self.old_encoder_cnn = None
        self.old_encoder_shallow = None
        self.old_decoder = None
        self._emb_mask_pre=None
        self._h_mask_pre=None
        self._c_mask_pre=None
        self._masks_back=None


    def _get_save_state(self, job, t, epoch, batch, save_optim, save_cnn):
        state = super()._get_save_state(job, t, epoch, batch, save_optim, save_cnn)
        state['emb_mask_pre'] = self._emb_mask_pre
        state['h_mask_pre'] = self._h_mask_pre
        state['c_mask_pre'] = self._c_mask_pre
        return state

    def load_model_state(self, model_dir, t=None, epoch=None, batch=None, load_optim=None, load_cnn=None, load_settings=False, verbose=False):
        state = super().load_model_state(model_dir, t, epoch, batch, load_optim, load_cnn, load_settings, verbose)
        self._emb_mask_pre = state['emb_mask_pre']
        self._h_mask_pre = state['h_mask_pre']
        self._c_mask_pre = state['c_mask_pre'] if 'c_mask_pre' in state.keys() else None
        return state

    def train_task(self, t: int, job: Job, start_ep=1, log_step=20, prev_task_epoch=None, monitor_metric='BLEU-4',
                   model_dir='models/', save_all_epochs=True, eval_all_epochs=True):
        if t > 0:
            #job_bleu_scores, _ = self.eval_job_bleu(job, sampling=True, verbose=True)

            # Activations mask for previous task t-1
            task = torch.tensor([t-1], dtype=torch.long).to(self.decoder.linear.weight.device)
            self.decoder: DecoderLSTMCellRATT
            emb_mask = self.decoder.hat_mask_emb.forward(task, self.settings.smax)
            h_mask = self.decoder.hat_mask_lstm.forward(task, self.settings.smax)
            c_mask = self._get_c_mask(self._get_active_words(job, t-1))

            if t == 1: # previous task for task t-1 for t==1 is task -1 --> task t-1 is the first task
                self._emb_mask_pre = emb_mask.detach()
                self._h_mask_pre = h_mask.detach() #.data.clone()
                self._c_mask_pre = c_mask.detach() #.data.clone()
            else:
                self._emb_mask_pre = torch.max(self._emb_mask_pre, emb_mask.detach())
                self._h_mask_pre = torch.max(self._h_mask_pre, h_mask.detach())
                self._c_mask_pre = torch.max(self._c_mask_pre, c_mask.detach())

            # Weights mask
            self._masks_back = {}
            for n, _ in self.decoder.named_parameters():
                vals = self.decoder.get_hat_view_for(n, self._emb_mask_pre, self._h_mask_pre, self._c_mask_pre)
                if vals is not None:
                    self._masks_back[n] = 1 - vals

        ret = super().train_task(t, job, start_ep, log_step, prev_task_epoch, monitor_metric, model_dir,
                                 save_all_epochs, eval_all_epochs)
        return ret

    def _get_c_mask(self, active_words):
        c_mask = torch.zeros_like(self.decoder.linear.weight[:, 0])
        c_mask[active_words] = 1
        return c_mask

    def _new_epoch_data(self, train_loader, epoch):
        return EpochData(train_loader, epoch, losses=('loss', 'cce', 'hat'), metrics=('top1', 'top5', 'bck%', 'fwd%', 'prev_fwd%' ))


    def _train_step(self, job, t, epoch, i, train_loader, images, targets, lengths, extra, active_words, epdata):
        smax = self.settings.smax
        s = (smax - 1/smax) * i/len(train_loader) + 1/smax
        self.decoder._current_train_s = s

        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)
        nb_words = lengths.sum()

        time_forward = time()

        # Forward
        features = self._encode(images)
        self.decoder.init_ratt_forward(t, s)
        logits, _, _, (emb_mask, h_mask) = self.decoder.forward(features, targets, lengths, active_words)
        #c_mask = self._get_c_mask(active_words)

        # Loss
        cce = self.criterion(logits[:, active_words], packed_targets.data)
        hat_reg = self.regularizer(emb_mask, h_mask)
        loss = cce + hat_reg
        epdata.upd_loss('cce', cce.detach().cpu(), nb_words)
        epdata.upd_loss('hat', hat_reg.detach().cpu(), nb_words)
        epdata.upd_loss('loss', loss.detach().cpu(), nb_words)
        epdata.upd_timer('forward', time_forward)

        time_metric = time()
        # Metrics etc..
        epdata.upd_metric('top5', accuracy(logits[:, active_words], packed_targets.data, 5), nb_words)
        epdata.upd_metric('top1', accuracy(logits[:, active_words], packed_targets.data, 1), nb_words)
        packed_preds = PackedSequence(logits[:, active_words].argmax(-1), packed_targets.batch_sizes,
                                      packed_targets.sorted_indices, packed_targets.unsorted_indices)

        bin_e = (emb_mask > .5).float()
        bin_h = (h_mask > .5).float()
        fwd_usage = (bin_e.sum() + bin_h.sum())/(bin_e.numel() + bin_h.numel())
        if t >= 1:
            bin_prev_e = (self._emb_mask_pre>.5).float()
            bin_prev_h = (self._h_mask_pre>.5).float()
            bck_e = (1-bin_prev_e)*bin_e
            bck_h = (1-bin_prev_h)*bin_h
            bck_e_free = (1-torch.max(bin_prev_e, bin_e))
            prev_fwd_usage = (bin_prev_e.sum() + bin_prev_h.sum()) / (bin_prev_e.numel() + bin_prev_h.numel())
            bck_usage = (bck_e.sum() + bck_h.sum()) / (bck_e.numel() + bck_h.numel())

        else:
            bck_usage = 1
            prev_fwd_usage = 0

        epdata.upd_metric('bck%', bck_usage*100, 1)
        epdata.upd_metric('fwd%', fwd_usage*100, 1)
        epdata.upd_metric('prev_fwd%', prev_fwd_usage*100, 1)

        epdata.store_pred_caps(*pad_packed_sequence(packed_preds, batch_first=True))
        epdata.store_targets(*pad_packed_sequence(packed_targets, batch_first=True))
        epdata.upd_timer('metric', time_metric)


        time_backward = time()
        # Backward and Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if self.settings.freeze_old_words and t>0 and self.old_words is not None:
            self.decoder_rnn.word_embed.weight.grad[self.old_words] = 0

        # Restrict layer gradients in backprop
        if t > 0:
            for n, p in self.decoder.named_parameters():
                if n in self._masks_back:
                    if self.settings.binary_backward_masks:
                        p.grad.data *= (self._masks_back[n] > 0.5).float()
                    else:
                        p.grad.data *= self._masks_back[n]

        # Compensate embedding gradients
        thres_cosh = self.settings.thres_cosh
        smax = self.settings.smax
        for n, p in self.decoder.named_parameters():
            if n.startswith('hat_mask'):
                num = torch.cosh(torch.clamp(s * p.data, -thres_cosh, thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den

        self.optimizer.step()
        epdata.upd_timer('backward', time_backward)

    def regularizer(self, emb_mask, h_mask) -> torch.Tensor:
        import numpy as np
        numerator = torch.tensor(0.).to(self.device)
        denominator = 0
        masks = [emb_mask, h_mask]
        if self._emb_mask_pre is not None and self._h_mask_pre is not None:
            masks_pre = [self._emb_mask_pre, self._h_mask_pre]
            for m, mp in zip(masks, masks_pre):
                aux = 1 - mp
                numerator += (m * aux).sum()
                denominator += aux.sum()
        else:
            for m in masks:
                numerator += m.sum()
                denominator += np.prod(m.size()).item()
        # numerator /= denominator
        # return self.settings.lambd * numerator
        return self.settings.lambd * (numerator / denominator)

    def evaluate(self, t, job: Job, data_loader, sampling=True):
        self.decoder.init_ratt_forward(t, self.settings.smax, self.settings.binary_sample_forward_masks)
        return super().evaluate(t, job, data_loader, sampling)

    def agnostic_evaluate(self, t, job: Job, data_loader, sampling=True):
        evals = {}
        for at in enumerate(len(job.tasks)):
            self.decoder.init_ratt_forward(at, self.settings.smax, self.settings.binary_sample_forward_masks)
            evals[at] = super().evaluate(t, job, data_loader, sampling)

    def agnostic_eval_job(self, job: Job, sampling=True, verbose=True, tasks=None):
        vprint = VPrinter(verbose)
        job_val_data = []
        tasks = [job.tasks[t] for t in tasks] if tasks is not None else job.tasks
        agnostic_epdata = EpochData()
        for val_t, val_task in enumerate(tasks):
            vprint(f"Evaluate task {val_t} ({val_task.name})")
            epdata = self.agnostic_evaluate(val_t, job, val_task.val_loader, sampling)
            for i in range(len(epdata[0].pred_caps_confidence)):
                lens = [epdata[at].pred_caps_len[i] for at in epdata.keys() ]
                min_len = min(lens)
                task_preds_prob = [torch.mean(epdata[at].pred_caps_confidence[i][:min_len]) for at in epdata.keys()]
                best_task = torch.argmax(task_preds_prob)
        return

    def agnostic_eval_job_metrics(self, job: Job, bleu=None, sampling=True, verbose=True, tasks=None):
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
