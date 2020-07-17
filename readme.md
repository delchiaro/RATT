# RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning
This repo contains the original code used for the experiments of 
[RATT: Recurrent Attention to Transient Tasks for Continual Image Captioning](https://openreview.net/forum?id=DlhyudbShm)
paper, and can be used to replicate the results.


## Abstract
>Research on continual learning has led to a variety of approaches to
mitigating catastrophic forgetting in feed-forward classification networks.
Until now surprisingly little attention has been focused on continual learning
of recurrent models applied to problems like image captioning. In this paper
we take a systematic look at continual learning of LSTM-based models for image
captioning. We propose an attention-based approach that explicitly
accommodates the *transient* nature of vocabularies in continual image
captioning tasks -- i.e. that task vocabularies are not disjoint. We call our
method Recurrent Attention to Transient Tasks (RATT), and also show how to
adapt continual learning approaches based on weight regularization and
knowledge distillation to recurrent continual learning problems. We apply our
approaches to incremental image captioning problem on two new continual
learning benchmarks we define using the MS-COCO and Flickr30 datasets. Our
results demonstrate that RATT is able to sequentially learn five captioning
tasks while incurring *no* forgetting of previously learned ones.

## Dependencies
This is the list of python requirements:
```
python==3.8.2
torch==1.4.0
torchvision==0.5.0
numpy==1.18.1
pandas==1.0.3
Pillow==7.2.0
h5py==2.10.0
matplotlib==3.1.3
seaborn==0.10.1
bidict==0.19.0
dacite==1.5.0
nltk==3.4.5
pycocotools==2.0.0
tqdm==4.43.0
attrs==19.3.0
attr==0.3.1
rouge_score==0.0.3
nlg_eval==2.3
dataclasses==0.7
```

On a common linux distribution like Ubuntu, we can create a working environment for this project following this 
procedure:
1. Create the conda environment with the provided ```environment.yml``` file:
    ```
    conda env create -f environment.yml
    ```  
2. Activate ratt environment: ```conda activate ratt```
3. Install the remaining packages manually:
   ```
   conda install pytorch==1.4.0
   pip install dacite==1.5.0 
   pip install rouge_score==0.0.3
   ```
4. Install nlg-eval following the instructions described 
[here](https://github.com/Maluuba/nlg-eval), i.e.:
    ```
    pip install git+https://github.com/Maluuba/nlg-eval.git@master
    conda install click
    nlg-eval --setup
    ```
  
#### nlg-eval
The installation process of nlg-eval is not straightforward, we advice to visit 
the original repo page here: https://github.com/Maluuba/nlg-eval

If you like to use use pycharm on a small laptop and run experiments on a remote 
interpreter located in a powerful server (like I use to do), you should read this 
section to correctly set up nlg-eval for the remote interpreter:
 * I assume you are using a conda environment with name *TRUE_ENV_NAME* in the remote server.
 * Install all the required packages for the current project. 
 * Create a new fake environment with name FAKE_ENV_NAME in your remote machine with a script 
 file  **python.sh** with permission 755+x with the following content:
``` 
    #!/bin/bash -l
    /CONDAPATH/envs/TRUE_ENV_NAME/bin/python "$@"
```
 
* Set the pycharm remote interpreter to:
```
    /CONDAPATH/envs/FAKE_ENV_NAME/bin/python.sh
```
* Now bash variables will be initialized on pycharm before each execution
   (and so java can be executed).
   



## Replicate Paper Experiments
We report here the command to be executed in order to replicate the paper experiments.
All the training experiments are executed with a fixed seed (42).
We ```--gpu 0``` parameter can be changed to use a different GPU index.

#### Dataset pre-processing
In order to train the models we have to pre-process the datasets.
We will use ResNet-152 pre-trained on ImageNet to extract features from 
all the images of the original dataset.
We will process each image only after resizing so that it will have 256 pixel on the 
shorter dimension, and after center-cropping a patch with size 224x224.

To pre-process MS-COCO:
1) Change the path to the MS-COCO dataset in the first line of ```coco_settings.py```
2) Run the script ```coco_feats.py```, you can use custom parameters if you like, e.g. ```--gpu```,
```--workers``` or ```-bs```.

To pre-process Flickr30k:
1) Change the path to the Flickr30k dataset in the first line of ```flickr30k_settings.py```
2) Run the script ```flickr30k_feats.py```.

#### MS-COCO Experiments
To train the models on MS-COCO with the proposed TASFI split:
```bash
    python train.py --gpu 0 --seed 42 -j coco-TASFI -ee 10 -mdl 26 -f coco_ft
    python train.py --gpu 0 --seed 42 -j coco-TASFI -ee 10 -mdl 26 -f coco_ewc -a ewc --ewc-sampling multinomial --ewc-lambda 10
    python train.py --gpu 0 --seed 42 -j coco-TASFI -ee 10 -mdl 26 -f coco_lwf -a lwf --lwf-T 1
    python train.py --gpu 0 --seed 42 -j coco-TASFI -ee 10 -mdl 26 -f coco_ratt -a ratt --ratt-usage 60 --ratt-smax 400 --seed 42 --gpu 0    

```

To evaluate on MS-COCO-TASFI test-set:
```bash
    python eval.py --gpu 0 --seed 42 -j coco-TASFI --test -f coco_ft
    python eval.py --gpu 0 --seed 42 -j coco-TASFI --test -f coco_ewc
    python eval.py --gpu 0 --seed 42 -j coco-TASFI --test -f coco_lwf
    python eval.py --gpu 0 --seed 42 -j coco-TASFI --test -f coco_ratt
```
#### Flickr30k Experiments
To train the models on Flickr30K with the proposed SAVI split:
```bash
    python train.py --gpu 0 --seed 42 -j flickr30k-SAVI -lr 1e-4 -bs 32 --nb-epochs 50 --extra_epochs 20 -mdl 40 -f flickr_ft 
    python train.py --gpu 0 --seed 42 -j flickr30k-SAVI -lr 1e-4 -bs 32 --nb-epochs 50 --extra_epochs 20 -mdl 40 -f flickr_ewc -a ewc --ewc-sampling multinomial --ewc-lambda 20
    python train.py --gpu 0 --seed 42 -j flickr30k-SAVI -lr 1e-4 -bs 32 --nb-epochs 50 --extra_epochs 20 -mdl 40 -f flickr_lwf -a lwf --lwf-T 1
    python train.py --gpu 0 --seed 42 -j flickr30k-SAVI -lr 1e-4 -bs 32 --nb-epochs 50 --extra_epochs 20 -mdl 40 -f flickr_ratt -a ratt --ratt-usage 60 --ratt-smax 400
```

Finally, to evaluate the trained models on Flickr30K-SAVI test set:    
```bash
    python eval.py -j flickr30k-SAVI --gpu 0 -f flickr_ft --test    
    python eval.py -j flickr30k-SAVI --gpu 0 -f flickr_ewc --test
    python eval.py -j flickr30k-SAVI --gpu 0 -f flickr_lwf --test
    python eval.py -j flickr30k-SAVI --gpu 0 -f flickr_ratt --test
    
```
 
 After training a folder per each model will be created into the ```model/``` folder.
 In each model folder you will find the weights of the model at the end of each training epoch,
 a result csv file per each task that report per-epoch performances over validation set,
 a result csv file with all aggregated epochs (results_all_[...].csv)and a result csv file with only the epoch
 until the best performing epoch over validation set respect to BLEU-4 sore (results_best_[...].csv).
 

   
## CLI and scripts
In this section we describe the main command-line tools and other scripts that can be used 
to run the experiments and evaluate trained models.

#### train.py
This can be used to train a model from scratch or to continue the training of an existing model.
Use ```-j JOB_NAME``` to select a job, ```-a APPROACH_NAME``` to select an approach from the available ones,
```-ne N``` to choose the number of training epochs per each task, ```-lr``` to select the
learning rate (e.g. ```--lr 1e-5```), ```--bs``` to select the batch size,
```--gpu X``` to run on the seleted GPU, etc..
Each technique has also a set of available aguments or flags starting with the technique name,
 that are simply ignored when used on a different technique. 

```
$ python train.py --help

Train a continual learning model for image captioning with different approaches.

 Optional arguments:
  -h, --help                    show this help message and exit
  -j, --job  {coco-TASFI,flickr30k-SAVI} Select the job name to use for the current experiment.
  --test                        Use test-set instead of validation set
  -bs BS                        Batch size to be used during training and evaluation.
  -f, --folder FOLDER           Model folder where to load/save weights and csv files.
  -a, --approach {ft, ewc, lwf, ratt, ratt_ablation}
  -t, --task TASK       Continue training the model from the selected task.
  -e, --epoch EPOCH     Continue training the model from the selected epoch, loading weights from previous one.If epoch 1 is chosen, best epoch of the previous task will be loaded.
  -l, --load LOAD       Load the best model weights from the first task of selected model/folder
  --hidden-size HIDDEN_SIZE     Number of neurons in LSTM hidden layer (hidden-state size)
  -emb-size EMB_SIZE            Number of neurons in image and word embedding layers (LSTM input size)
  -mdl, --max-decode-len MDL    Max decoding lenght for sampling (evaluation)
  -ne, --nb-epochs NB_EPOCHS    Number of training epoch to run on each task.
  -ex, --examples EXAMPLES      Number of examples to use in each task during training, useful to speedup debugging.
  -ee, --extra-epochs EXTRA     Extra epochs for the first task.
  -lr LR                        Learning rate for Adam optimization algorithm
  -wd WD                        Weight decay regularization.
  --freeze-old-words            Prevent words to be trained in current task when they appeared in one of the previous tasks
  --ewc-sampling {true,max_pred,multinomial}
  --ewc-teacher-forcing Enable teacher forcing when computing fisher matrix
  --ewc-lambda EWC_LAMBDA       Loss multiplier applied to EWC loss
  --lwf-lambda LWF_LAMBDA       Loss multiplier applied to LwF loss
  --lwf-T LWF_T                 Temperature for LwF loss
  --lwf-h-distill               Distill hidden state together with output predictions
  --lwf-h-lambda LWF_H_LAMBDA
                                Loss multiplier applied to hidden state LwF loss, when --lwf-h-distill is enabled
  --ratt-lambda RATT_LAMBDA     Loss multiplier applied to RATT loss
  --ratt-thres-cosh RATT_THRES_COSH
  --ratt-smax RATT_SMAX         Maximum value for scaling parameter s.
  --ratt-usage RATT_USAGE       Network usage at the beginning of the train task.
  --ratt-bin-backward           Binarize RATT bacwkard masks.
  --ratt-bin-forward            Binarize RATT forwward masks.
  --ratt-emb                    Enable masks for Embedding layers when executing RATT ablation
  --ratt-cls                    Enable masks for classifier layers when executing RATT ablation
  -s SEED, --seed SEED          Chose the seed for current experiment
  -w WORKERS, --workers WORKERS Number of workers for dataloader
  -g GPU, --gpu GPU             GPU to be used from CUDA
  --threads THREADS             Number of threads that torch will be able to use.
  --pin                         Pin GPU memory.
```
#### eval.py

```
$ python eval.py --help

Evaluate a pre-trained continual learning model for image captioning.

optional arguments:
  -h, --help                    show this help message and exit
  -j, --job {coco-TASFI,flickr30k-SAVI}     Select the job name to use for the current experiment.
  --test                            Use test-set instead of validation set
  -bs BS                            Batch size to be used during training and evaluation.
  -f, --folder FOLDER [FOLDER ...]  Model folders where to load weights from.
  -o, --out OUT                     Output file name. If not specified the model folder name will be used.
  -t, --task TASK [TASK ...]        Evaluate the model loading weights related to selected tasks. Use -1 (default) to load the last task.
  -e, --epoch EPOCH [EPOCH ...]     Evaluate model loading weights at the selected epochs of the selected task. Use -1 (default) to load weights at best validation epoch.
  --ratt-bin-forward                Force binarization of forward masks for RATT approach
  -s, --seed SEED                   Chose the seed for current experiment
  -w, --workers WORKERS             Number of workers for dataloader
  -g, --gpu GPU                     GPU to be used from CUDA
  --threads THREADS                 Number of threads that torch will be able to use.
  --pin                             Pin GPU memory.

```


#### plots.py
This script can be used to generate some of the plots showed in the paper, but it's 
not a command-line tool: code should be modified to generate the correct plot for the
correct model.

#### coco_resize.py
This command line tool can be used to resize all the images in MS-COCO dataset
saving jpeg version of each resized image into a new directory.
The default size is defined in coco_settings.py (256), parameter ```--val``` is needed
when we want to process MS-COCO validation set instead of the training set.
This tool is not really needed anymore, because ```coco_feats.py``` can directly
read the original jpeg images and resize on the fly before processing with the CNN.
You should use the flag ```--resized-path``` on ```coco_feats.py``` if you want to process
 the images already resized with ```coco_resize.py```. 
 
 
