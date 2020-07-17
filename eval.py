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

load_tasks = [len(job.tasks) - 1 if tsk == -1 else tsk for tsk in eval_args.task]
load_epochs = ['best' if ep == -1 else ep for ep in eval_args.epoch]


data = {}

first_row = True
for folder in eval_args.folder:
    model_dir = f'models/{folder}'
    data[folder] = {}
    for t in load_tasks:
        task_name = job.tasks[t].name
        data[folder][task_name] = {}
        #data[folder][t] = data[folder][task_name]

        for ep in load_epochs:
            path = model_dir if t is None else Approach._model_state_path(model_dir, t, ep)
            print(f'\n\n\n\nLoading model settings {path}')
            print(f'Evaluating model <{folder}> after trained on task <{t}: {task_name}> after epoch <{ep}>')
            states = torch.load(path, map_location=device)
            train_args_dict = from_dict(Approach.Settings, states['settings']).args
            for train_arg_key in train_args_dict.keys():
                if hasattr(eval_args, train_arg_key):
                    train_args_dict[train_arg_key] = eval_args.__dict__[train_arg_key]
            train_args = Namespace(**train_args_dict)
            appr = init_approach(train_args, job, cnn_feats_size, device)
            appr.load_model_state(model_dir, t, ep, load_settings=True)

            scores, epochdata = appr.eval_job_metrics(job)
            data[folder][task_name][ep] = {** {job.tasks[i].name: scores[i] for i in range(len(scores))},
                                           #**{i: scores[i] for i in range(len(scores))}
                                           }
metrics = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE_L', 'CIDEr']
tasks = data[folder][task_name][ep].keys()
metrics = list(scores.keys()) if ( isinstance(metrics, str) and metrics.lower() == '__all__') else metrics
for metric in metrics:
    assert metric in scores[0].keys()







#%%

def write_csv_results_rowtech(out_file, data, metrics):
    out_csv = open(out_file, 'w')
    header1 = ', '.join(['', '', ''] + [t for tl in [[task] * len(metrics) for task in tasks] for t in tl])
    header2 = ', '.join(['model-name', 'trained-task', 'epoch'] + metrics * len(tasks))
    out_csv.write(header1 + '\n')
    out_csv.write(header2 + '\n')
    for model_name, model_data in data.items():
        for trained_task, trtask_data in model_data.items():
            for epoch, epoch_trtask_data in trtask_data.items():
                line = f"{model_name}, {trained_task}, {epoch}"
                for eval_task, evtask_epoch_trtask_data in epoch_trtask_data.items():
                    line += ', ' + ', '.join(str(evtask_epoch_trtask_data[metric]) for metric in metrics)
                out_csv.write(line + "\n")
    out_csv.close()


def write_csv_results_rowmetrics(out_file, data, metrics):
    out_csv = open(out_file, 'w')
    headers1 = []
    headers2 = []
    all_columns = []
    for model_name, model_data in data.items():
        for trained_task, trtask_data in model_data.items():
            for epoch, epoch_trtask_data in trtask_data.items():
                header1 = []
                header2 = []
                columns = []
                for eval_task, evtask_epoch_trtask_data in epoch_trtask_data.items():
                    header1.append(f"{model_name} - {trained_task}@{epoch}")
                    header2.append([metric for metric in metrics])
                    columns.append([str(evtask_epoch_trtask_data[metric]) for metric in metrics])
                headers2.append(header1)
                headers1.append(header2)
                all_columns.append(columns)

    out_csv.write(", ")
    for block in range(len(all_columns[0])):
        for col in range(len(all_columns)):
            out_csv.write(f"{headers2[col][block]}, ")
    out_csv.write("\n")
    out_csv.write(", ")
    for block in range(len(all_columns[0])):
        for col in range(len(all_columns)):
            out_csv.write(f"{headers1[col][block]}, ")
    out_csv.write("\n")

    for row in range(len(all_columns[0][0])):
        out_csv.write(f"{metrics[row]}, ")
        for block in range(len(all_columns[0])):
            for col in range(len(all_columns)):
                out_csv.write(f"{all_columns[col][block][row],}")
        out_csv.write("\n")
    out_csv.write("\n")
    out_csv.close()

#%%


# EACH ROW IS A TECHNIQUE (FOLDER), EACH BLOCK OF COLUMNS IS A TEST-TASK, EACH COLUMN IN THE BLOCK IS A METRIC
out_fname = 'eval_rowtech_test_' if eval_args.test else 'eval_rowtech_val_'
out_fname += f"{'_'.join(eval_args.folder)}" if eval_args.out is None else eval_args.out
out_file = f"models/{out_fname}.csv"
write_csv_results_rowtech(out_file, data, metrics)


# TODO: this output is not working properly, eventually should be fixed.
# # EACH ROW IS A METRIC, EACH BLOCK OF COLUMNS IS A TEST-TASK, EACH COLUMN IN THE BLOCK IS A TECHNIQUE (FOLDER)
# out_fname = 'eval_rowmetrics_test_' if eval_args.test else 'eval_rowmetrics_val_'
# out_fname += f"{'_'.join(eval_args.folder)}" if eval_args.out is None else eval_args.out
# out_file = f"models/{out_fname}.csv"
# write_csv_results_rowmetrics(out_file, data, metrics)


