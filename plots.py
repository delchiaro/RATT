import sys
import matplotlib
from attr import dataclass
from typing import Dict, List, Optional

sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
os.getcwd()


@dataclass
class Experiment:
    ds_name: str
    results_dir: str
    method_folder_map: Dict[str, str]
    metrics: List[str]
    method_key: str = 'method'
    df_best_epochs: Optional[List[pd.DataFrame]] = None
    df_all_best_epochs: Optional[List[pd.DataFrame]] = None
    df_all_epochs: Optional[List[pd.DataFrame]] = None
    ds_tasks: Optional[Dict] = None

    @property
    def methods(self):
        return self.method_folder_map.keys()

    def read_files(self):
        self.read_best_epochs_files().read_all_epochs_files()
        return self

    def read_best_epochs_files(self):
        df_all_best_epochs = []
        df_best_epochs = []
        ds_tasks = {}
        print('Dataset:', self.ds_name)
        for method_name, dir in self.method_folder_map.items():
            print('Experiments:', method_name, ' directory:', dir)
            best_file = f'{self.results_dir}/{dir}/results_best_BLEU-1_BLEU-2_BLEU-3_BLEU-4_METEOR_ROUGE_L_CIDEr_TOP-1_TOP-5.csv'
            df = pd.read_csv(best_file)
            idx_of_second_header = df[df['training-task'] == 'training-task'].index.values[0]
            _all_ep_df = df[df.index.values < idx_of_second_header].copy()
            _all_ep_df['current-epoch'] = np.arange(len(_all_ep_df))
            df = df[df.index.values > idx_of_second_header]
            tasks = df['training-task'].unique()
            ds_tasks[self.ds_name] = tasks
            df['current_epoch'] = 1
            _df_all_ep = make_tidy_df(_all_ep_df, tasks, self.metrics)
            _df_best_ep = make_tidy_df(df, tasks, self.metrics)
            _df_all_ep[self.method_key] = method_name
            _df_best_ep[self.method_key] = method_name
            _df_all_ep['dataset'] = self.ds_name
            _df_best_ep['dataset'] = self.ds_name
            df_best_epochs.append(_df_all_ep)
            df_all_best_epochs.append(_df_best_ep)
        df_best_epochs = pd.concat(df_best_epochs)
        df_all_best_epochs = pd.concat(df_all_best_epochs)

        df_all_best_epochs.sort_values(['dataset', self.method_key, 'epoch'], inplace=True)
        df_best_epochs.head(10)

        self.df_all_best_epochs = df_all_best_epochs
        self.df_best_epochs = df_best_epochs
        self.ds_tasks = ds_tasks
        #return df_all_best_epochs, df_best_epochs, ds_tasks
        return self

    def read_all_epochs_files(self):
        df_all_epochs = []
        ds_tasks = {}
        print('Dataset:', self.ds_name)
        for method_name, dir in self.method_folder_map.items():
            print('Experiments:', method_name, ' directory:', dir)
            all_file = f'{self.results_dir}/{dir}/results_all_BLEU-1_BLEU-2_BLEU-3_BLEU-4_METEOR_ROUGE_L_CIDEr_TOP-1_TOP-5.csv'
            df = pd.read_csv(all_file)
            df['current-epoch'] = np.arange(len(df))
            tasks = df['training-task'].unique()
            ds_tasks[self.ds_name] = tasks
            # df['current_epoch'] = 1
            df = make_tidy_df(df, tasks, self.metrics)
            df[self.method_key] = method_name
            df['dataset'] = self.ds_name
            df_all_epochs.append(df)
        df_all_epochs = pd.concat(df_all_epochs)
        self.df_all_epochs = df_all_epochs
        self.ds_tasks = ds_tasks
        # return df_all_epochs, ds_tasks
        return self


def make_tidy_df(df, tasks, metrics):
    cols1 = [f'{t}-{m}' for t in tasks for m in metrics]
    cols2 = ['training-task', 'current-epoch']
    agg_df = df[cols1 + cols2]
    for c in cols1:
        agg_df.loc[:, c] = agg_df[c].astype('float')

    agg_df2 = agg_df.reset_index()
    agg_df2 = agg_df2.melt(id_vars=cols2, value_vars=cols1, value_name='score')
    agg_df2['eval task'] = agg_df2['variable'].map(lambda s: s[:s.index('-')])
    agg_df2['metric'] = agg_df2['variable'].map(lambda s: s[s.index('-') + 1:])
    agg_df2.rename(columns={cols2[0]: "task", cols2[1]: "epoch"}, inplace=True)
    del agg_df2['variable']
    return agg_df2


def epoch_plot(experiment: Experiment, fname=None,
               top=None, bottom=None, left=None, right=None, wspace=None, dpi=300.,
               min_y=None, max_y=None):
    def _plot(x, y, t, m, **kwargs):
        ds_name = kwargs['ds_name']
        ax = kwargs.get("ax", plt.gca())
        ax.plot(x, y)
        if min_y is not None or max_y is not None:
            ax.set_ylim(bottom=min_y, top=max_y)
        all_tasks = experiment.ds_tasks[ds_name]
        c = [(t == _t).sum() for _t in experiment.ds_tasks[ds_name]]
        ax.set_xticks(np.cumsum([0] + c[:-1]))
        word_idx = 1 if ds_name == 'Flickr30k' else 0
        ax.set_xticklabels([task_name[word_idx].upper() for task_name in all_tasks])

    from matplotlib import lines

    plt.style.use('default')
    g = sns.FacetGrid(experiment.df_all_epochs[experiment.df_all_epochs.dataset == experiment.ds_name],
                      sharex=False, legend_out=True, sharey=False,
                      row='metric', col="eval task", hue=experiment.method_key,
                      row_order=experiment.metrics, hue_order=experiment.methods,
                      margin_titles=True,
                      gridspec_kws={"top": top, "bottom": bottom, "left": left, "right": right})
    ax = g.map(_plot,'epoch', 'score', 'task', 'metric', ds_name=experiment.ds_name, legend_out=True)
    # There is no labels, need to define the labels

    # Create the legend patches
    legend_labels = experiment.methods
    lines_colors = [line.get_color() for line in ax.fig.get_children()[1].get_children() if isinstance(line, matplotlib.lines.Line2D)]
    legend_patches = [matplotlib.patches.Patch(color=C, label=L) for C, L in zip(lines_colors, legend_labels)]
    # Plot the legend
    g.add_legend(dict(zip(legend_labels, legend_patches)))
    #g.fig.suptitle(f'Dataset: {ds_name}', size=20)

    g.fig.set_dpi(dpi)
    g.fig.subplots_adjust(wspace=wspace)
    if fname is not None:
        g.fig.savefig(fname)
    plt.show()



def plot_heatmap(experiment: Experiment, fname=None,
                 top=None, bottom=None, left=None, right=None,
                 cmap=True, vmin=0, vmax=1.0, val_mul=100,
                 annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=200., wspace=None):
    def _plot_heatmap(x, y, v, **kwargs):
        ax = kwargs.get("ax", plt.gca())
        ds_name = kwargs['ds_name']
        tasks = experiment.ds_tasks[ds_name].tolist()
        n = len(tasks)
        m = np.zeros((len(tasks), len(tasks)))
        for t, et, v in zip(x, y, v):
            m[n-1-tasks.index(et), tasks.index(t)] = v
        sns.heatmap(m*val_mul, square=True, annot=annot, fmt=annot_fmt, annot_kws={"size": annot_size},
                    cbar=cmap, vmin=vmin*val_mul, vmax=vmax*val_mul)
        word_idx = 1 if ds_name == 'Flickr30k' else 0
        ax.set_xticklabels([t[word_idx].upper() for t in tasks])
        ax.set_yticklabels([t[word_idx].upper() for t in tasks][::-1])  # , rotation=90)

    plt.style.use('default')
    sns.set(font_scale=font_scale)
    g = sns.FacetGrid(experiment.df_best_epochs[experiment.df_best_epochs.dataset == experiment.ds_name],
                      #row='metric',
                      col=experiment.method_key,
                      margin_titles=True, row_order=experiment.metrics, col_order=experiment.methods,
                      sharex=True, legend_out=True,
                      gridspec_kws={"top": top, "bottom": bottom,
                                    "left": left, "right": right}
                      )
    g.fig.set_dpi(dpi)
    g.fig.subplots_adjust(wspace=wspace)

    g.map(_plot_heatmap, 'task', 'eval task', 'score', ds_name=experiment.ds_name)
    g.add_legend()
    for ax in g.axes.flat:
        for label in ax.get_yticklabels():
            label.set_rotation(0)
    if fname is not None:
        g.fig.savefig(fname)
    plt.show()



# ALL_RESULTS_DIR = './models/all_csvs/'
ALL_RESULTS_DIR = './models/'
RATT_ABLATION_RESULTS_DIR = './models/all_ratt_ablation_csv/'
SMAX_ABLATION_RESULTS_DIR = './models_ratt_smax_ablation_origparams/'

METRICS = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr', ]
METRICS = ['BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr', ]
METRICS = ['BLEU-4' ]
           #'TOP-1', 'TOP-5']

FLICKR = 'Flickr30k'
COCO = 'MS-COCO14'

#%% METHODS COMPARISON PLOTS

exp_flickr_comparison = Experiment(FLICKR, ALL_RESULTS_DIR,
                                   {'FT': 'flickr30k-SAVI_bs32_lr1e-4_ft',
                                    'EWC': 'flickr30k-SAVI_bs32_lr1e-4_ewc_multi_l20',
                                    'LwF': 'flickr30k-SAVI_bs32_lr1e-4_lwf_T1_l2',
                                    'RATT': 'flickr30k-SAVI_bs32_lr1e-4_ratt_u60_s2000_l5'},
                                    metrics=['BLEU-4']).read_files()
exp_coco_comparison = Experiment(COCO, ALL_RESULTS_DIR,
                                   {'FT': 'coco_tasfi_ft',
                                    'EWC': 'coco_tasfi_ewc_multi_10',
                                    'LwF': 'coco_tasfi_lwf_T1',
                                    'RATT': 'coco_tasfi_ratt_60_400'},
                                    metrics=['BLEU-4']).read_files()

epoch_plot(exp_flickr_comparison, 'plots/flickr_comparison_epochs', left=0.06, right=.9, bottom=.2)

plot_heatmap(exp_flickr_comparison, 'plots/flickr_comparison_heatmaps',
             top=0.9, bottom=0.2, left=0.05, right=.98,
             cmap=True, vmin=0, vmax=.25, val_mul=100,
             annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=300.)


epoch_plot(exp_coco_comparison, 'plots/coco_comparison_epochs', left=0.06, right=.9, bottom=.2)

plot_heatmap(exp_coco_comparison, 'plots/coco_comparison_heatmaps',
             top=0.9, bottom=0.2, left=0.05, right=.98,
             cmap=True, vmin=0, vmax=.32, val_mul=100,
             annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=300., wspace=0.1)


#%% RATT ABLATIONS PLOTS

exp_mscoco_ratt_ablation = Experiment(COCO, RATT_ABLATION_RESULTS_DIR,
                                      {'FT': '1_coco_tasfi_ft',
                                      'LSTM': '2_coco_tasfi_hatabl',
                                      'LSTM+CLS': '3_coco_tasfi_hatabl_cls',
                                      'LSTM+EMB': '4_coco_tasfi_hatabl_emb',
                                      'RATT': '5_coco_tasfi_ratt_60_40'},
                                      metrics=['BLEU-4']).read_files()
#%%
epoch_plot(exp_mscoco_ratt_ablation, 'plots/coco_ratt_ablation_epochs', left=0.04, right=.88, bottom=.15)
#%%
plot_heatmap(exp_mscoco_ratt_ablation, 'plots/coco_ratt_ablation_heatmaps',
             top=0.9, bottom=0.2, left=0.05, right=.98,
             cmap=True, vmin=0, vmax=.32, val_mul=100,
             annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=300., wspace=0.1)
#%% SMAX ABLATIONS PLOTS

exp_mscoco_smax_ablation = Experiment(COCO, SMAX_ABLATION_RESULTS_DIR,
                                      {**{'FT': 'coco_tasfi_ft'},
                                       **{f'{smax}': f'coco_smax{smax}_lambd5000_usage_60_seed_42'
                                          for smax in [50, 250, 400, 1000]}},
                                      metrics=['BLEU-4'],
                                      method_key='s_max').read_files()

exp_flickr_smax_ablation = Experiment(FLICKR, SMAX_ABLATION_RESULTS_DIR,
                                      {**{'FT': 'flickr_savi_ft'},
                                       **{f'{smax}': f'flickr_smax{smax}_lambd5_usage_60_seed_42'
                                          for smax in [#10, 100
                                                       50, 400, 2000, 6000]}},
                                      metrics=['BLEU-4'],
                                      method_key='s_max').read_files()
#%%
epoch_plot(exp_mscoco_smax_ablation, 'plots/coco_smax_ablation_epochs', left=0.04, right=.91, bottom=.15)
plot_heatmap(exp_mscoco_smax_ablation, 'plots/coco_smax_ablation_heatmaps',
             top=0.9, bottom=0.2, left=0.05, right=.98,
             cmap=True, vmin=0, vmax=.32, val_mul=100,
             annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=300., wspace=0.1)
#%%
epoch_plot(exp_flickr_smax_ablation, 'plots/flickr_smax_ablation_epochs', left=0.06, right=.90, bottom=.15)
plot_heatmap(exp_flickr_smax_ablation, 'plots/flickr_smax_ablation_heatmaps',
             top=0.9, bottom=0.2, left=0.05, right=.98,
             cmap=True, vmin=0, vmax=.32, val_mul=100,
             annot=True, annot_fmt='.1f', annot_size=11, font_scale=1.3, dpi=300., wspace=0.1)

