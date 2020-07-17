from os.path import join

import torch
from torchvision.transforms import transforms

from init import parse_args, init_job, init_approach, init_torch

#### Images Parameters
crop_size = 224
resize_size = 256
cnn_feats_size = 2048

train_transform = transforms.Compose([transforms.Resize(resize_size),
                                      transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
val_transform = transforms.Compose([transforms.Resize(resize_size),
                                    transforms.CenterCrop(crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#### Argument Parsing
eval_args = parse_args(eval=True)
device = init_torch(eval_args)


#%%
from CNIC.approach import Approach
from approaches.ewc import ApproachEWC
from dacite import from_dict
from argparse import Namespace


#### Paths
job = init_job(eval_args, train_transform, val_transform)

if not isinstance(eval_args.folder, list):
    eval_args.folder = [eval_args.folder]
if not isinstance(eval_args.task, list):
    eval_args.task = [eval_args.task]
if not isinstance(eval_args.epoch, list):
    eval_args.epoch = [eval_args.epoch]

load_tasks = range(len(job.tasks)) if eval_args.task == [-1] else eval_args.task
load_epochs = ['best' if ep == -1 else ep for ep in eval_args.epoch]


data_scores = {}
data_epochs = {}

first_row = True
for folder in eval_args.folder:
    model_dir = f'models/{folder}'
    data_scores[folder] = {}
    data_epochs[folder] = {}
    for tt in load_tasks:
        task_name = job.tasks[tt].name
        data_scores[folder][tt] = {}
        data_epochs[folder][tt] = {}
        #data[folder][t] = data[folder][task_name]
        for ep in load_epochs:
            path = model_dir if tt is None else Approach._model_state_path(model_dir, tt, ep)
            print(f'\n\n\n\nLoading model settings {path}')
            print(f'Evaluating model <{folder}> after trained on task <{tt}: {task_name}> after epoch <{ep}>')
            states = torch.load(path, map_location=device)
            train_args_dict = from_dict(Approach.Settings, states['settings']).args
            for train_arg_key in train_args_dict.keys():
                if hasattr(eval_args, train_arg_key):
                    train_args_dict[train_arg_key] = eval_args.__dict__[train_arg_key]
            train_args = Namespace(**train_args_dict)
            appr = init_approach(train_args, job, cnn_feats_size, device)
            appr.load_model_state(model_dir, tt, ep, load_settings=True)

            scores, epochdata = appr.eval_job_metrics(job)

            data_scores[folder][tt][ep] = {** {job.tasks[i].name: scores[i] for i in range(len(scores))},
                                                  #**{i: scores[i] for i in range(len(scores))}
                                                  }
            data_epochs[folder][tt][ep] = epochdata

#%%

# coco_tasfi_ft
# coco_tasfi_ewc_multi_10
# coco_tasfi_lwf_T1
# coco_tasfi_hat_60_400
#eval python eval.py -j coco-TASFIv2 -f coco_tasfi_ft coco_tasfi_ewc_multi_10 coco_tasfi_lwf_T1 coco_tasfi_hat_60_400 -t 4 --test --gpu 0


#%%
all_t = [0, 1, 2, 3, 4]
targets = {}
preds = {}
extra = {}

models = []
task_names = []
for model_name in data_epochs.keys():
    models.append(model_name)
    task_names = []
    targets[model_name] = {}
    preds[model_name] = {}
    extra[model_name] = {}
    for tt in data_epochs[model_name].keys():
        task_names.append(job.tasks[tt].name)
        task_epdata = data_epochs[model_name][tt]['best']
        # task_epdata = data_epochs[model_name]['interior']['best']
        targets[model_name][tt] = {}
        preds[model_name][tt] = {}
        extra[model_name][tt] = {}

        for t in all_t:
            targets[model_name][tt][t] = [job.tasks[t].vocab.translate_all(target_cap_batch) for target_cap_batch in task_epdata[t].get_target_caps()]
            preds[model_name][tt][t] = job.tasks[t].vocab.translate_all(task_epdata[t].get_pred_caps())
            extra[model_name][tt][t] = task_epdata[t].extra_data
#%%

def print_preds(D, train_task, eval_task, example_idx, bs=128):
    from pprint import pprint
    targets = D['targets']
    preds = D['preds']
    extra = D['extra']
    models = list(preds.keys())
    if isinstance(train_task, int):
        train_task = [train_task]
        # for model in models:
        #     print(f"{model} pred:")
        #     print(preds[model][train_task][eval_task][example_idx], end="\n\n")
        # print("Targets: ")
        # pprint(targets[models[1]][train_task][eval_task][example_idx])

    for model in models:
        print(f"{model} pred:")
        for t in train_task:
            print(f'Trained on {t}: {preds[model][t][eval_task][example_idx]}')
        print('\n')

    print("Targets: ")
    pprint(targets[models[1]][train_task[0]][eval_task][example_idx])
    # extra[models[1]][eval_task][pred_idx]

    import matplotlib.pyplot as plt
    fname = extra[models[1]][train_task[0]][eval_task][example_idx//bs][example_idx%bs]['file_name'].decode('utf8')
    im = plt.imread(f'/equilibrium/delchiaro/datasets/mscoco14/coco_images/val2014/{fname}')
    plt.imshow(im)
    plt.show()


#%% SAVE ON PICKLE
import pickle, os
train_task = 0
train_task = load_tasks[0] if len(load_tasks) == 1 else train_task
D = {'preds': preds,
     'targets': targets,
     'extra': extra}
os.makedirs('caption_images_data/', exist_ok=True)
with open(f'caption_images_data/preds_{job.name}_t_{train_task}.pkl', 'wb') as fname:
    pickle.dump(D, fname)


#%% LOAD FROM PICKLE

import collections

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

import pickle, os
job_name = "coco-TASFIv2.job.pkl"
folders = [f for f in os.listdir('caption_images_data/') if f.startswith(f"preds_{job_name}")]
Ds = []
for fname in folders:
    f_tt = int(fname.split('_t_')[1].split('.pkl')[0])
    with open(f'caption_images_data/{fname}', 'rb') as f:
        Ds.append(pickle.load(f))
D = {}
D = Ds[0]
for d in Ds:
    dict_merge(D, d)

load_tasks = [0]

#%% SHOW RESULTS
train_task = 0
train_task = load_tasks[0] if len(load_tasks) == 1 else train_task
print_preds(D, train_task, eval_task=0, example_idx=301, bs=128)

#%%
train_task = 0
train_task = load_tasks[0] if len(load_tasks) == 1 else train_task
print_preds(D, train_task, eval_task=0, example_idx=301, bs=128)


#%%
train_task = load_tasks[0] if len(load_tasks) == 1 else train_task
print_preds(D, train_task=[0, 1, 2, 3, 4], eval_task=1, example_idx=111, bs=128)





#%%
print_preds(D, train_task=[0, 1, 2, 3, 4], eval_task=3, example_idx=10, bs=128)

#%%
train_task = load_tasks[0] if len(load_tasks) == 1 else train_task
print_preds(D, train_task=[0, 1, 2, 3, 4], eval_task=3, example_idx=10, bs=128)

