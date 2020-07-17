from typing import Dict
import time
from CNIC.utils import AverageMeter


class EpochData:
    def __init__(self, loader=None, epoch_number=None, timers=('data', 'forward', 'metric', 'backward'),
                 losses=('loss', ), metrics=('top1', 'top5')):
        self.epoch_number = epoch_number
        self.loader = loader

        self.pred_caps = []
        self.pred_caps_len = []
        self.target_caps = []
        self.target_caps_len = []
        self.extra_data = []

        self.timers: Dict[str, AverageMeter] = {n: AverageMeter(print_value=False) for n in timers}
        self.timers['batch'] = AverageMeter(print_value=True)
        self.losses: Dict[str, AverageMeter] = {n: AverageMeter() for n in losses}
        self.metrics: Dict[str, AverageMeter] = {n: AverageMeter() for n in metrics}

        self.epoch_elapsed_time = 0.

    def timer(self, name: str)  -> AverageMeter:
        return self.timers[name]

    def loss(self, name: str) -> AverageMeter:
        return self.losses[name]

    def metric(self, name: str) -> AverageMeter:
        return self.metrics[name]

    def add_timer(self, name: str, print_value=False, print_avg=True):
        if name not in self.timers.keys():
            self.timers[name] = AverageMeter(print_value, print_avg)
        return self

    def add_loss(self, name: str, print_value=True, print_avg=True):
        if name not in self.timers.keys():
            self.losses[name] = AverageMeter(print_value, print_avg)
        return self

    def add_metric(self, name: str, print_value=True, print_avg=True):
        if name not in self.timers.keys():
            self.metrics[name] = AverageMeter(print_value, print_avg)
        return self

    def upd_timer(self, name, initial_time, final_time=None):
        final_time = time.time() if final_time is None else final_time
        self.timer(name).update(final_time-initial_time)
        return self

    def upd_loss(self, name, value, n=1):
        self.loss(name).update(value, n)
        return self

    def upd_metric(self, name, value, n=1):
        self.metric(name).update(value, n)
        return self

    def store_pred_caps(self, pred_caps, pred_cap_lens):
        self.pred_caps.append(pred_caps.cpu())
        self.pred_caps_len.append(pred_cap_lens.detach().cpu())

    def store_targets(self, target_caps, target_cap_lens):
        self.target_caps.append(target_caps.cpu())
        self.target_caps_len.append(target_cap_lens.detach().cpu())

    def __str__(self):
        return self.__epoch_str() + self.__status_str()

    def __epoch_str(self):
        return f"Epoch: [{self.epoch_number}]\t" if self.epoch_number is not None else ""

    def __timers_str(self, all_timers=True):
        str = f"Batch time: {self.timer('batch')}"
        if all_timers:
            str += " " + "[" + ", ".join([f"{n}: {t}" for n, t in self.timers.items() if n != 'batch']) + "]"
        return str

    def __losses_str(self):
        return ", ".join([f"{n}: {l.str(fp=4)}" for n, l in self.losses.items()])

    def __metrics_str(self):
        return ", ".join([f"{n}: {m.str(fp=2)}" for n, m in self.metrics.items()])

    def __status_str(self, all_timers=True):
        return " ".join([self.__timers_str(all_timers), self.__losses_str(), self.__metrics_str()])


    def print_batch_status(self, batch_number=None, all_timers=True):
        str = self.__epoch_str()
        if batch_number is not None:
            str += f"[{batch_number}" + f"/{len(self.loader)}]\t" if self.loader is not None else "]\t"
        str += self.__status_str(all_timers)
        print(str)

    def get_target_caps(self, cpi=None):
        if cpi is None:
            cpi = self.target_caps[0].shape[0] // self.pred_caps[0].shape[0]
        zipped_targets = zip(self.target_caps, self.target_caps_len)
        tolist = lambda x: x.numpy().tolist()
        targets = [tolist(cap)[1:len - 1] for caps, lengths in zipped_targets for cap, len in zip(caps, lengths)]
        grouped_targets = [targets[i * cpi:((i * cpi) + cpi)] for i in range(len(targets) // cpi)]
        return grouped_targets


    def get_pred_caps(self):
        tolist = lambda x: x.detach().numpy().tolist()
        zipped_preds = zip(self.pred_caps, self.pred_caps_len)
        preds = [tolist(cap)[1:len - 1] for caps, lengths in zipped_preds for cap, len in zip(caps, lengths)]
        return preds