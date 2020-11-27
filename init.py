import argparse
import sys
from datetime import datetime

import h5py

from CNIC.approach import Approach
from CNIC.model import EncoderCNN, DecoderLSTMCell
from CNIC.task import JobFactory
from approaches.ewc import ApproachEWC
from approaches.ratt import ApproachRATT, DecoderLSTMCellRATT
from approaches.ratt_ablation import ApproachRATTAblation, DecoderLSTMCellRATTAblation
from approaches.lwf30 import ApproachLwF_3
import jobs


# NB_TRAIN_EXAMPLES = NB_VAL_EXAMPLES = None # 200
DEBUGGING = sys.gettrace() is not None
jobs_factories = {job.name: job for _, job in jobs.__dict__.items() if isinstance(job, JobFactory)}
default_job_name = jobs.coco_TASFI.name


def parse_args(eval=False):
    train_descr = 'Train a continual learning model for image captioning with different approaches.'
    eval_descr = 'Evaluate a pre-trained continual learning model for image captioning.'

    parser = argparse.ArgumentParser(description=eval_descr if eval else train_descr)
    parser.add_argument('-j', '--job', type=str, default=default_job_name, choices=jobs_factories.keys(),
                        help='Select the job name to use for the current experiment.')
    parser.add_argument('--test', action='store_true', default=False, help='Use test-set instead of validation set')

    parser.add_argument('-bs', type=int, default=128, help='Batch size to be used during training and evaluation.')

    if eval:
        parser.add_argument('-f', '--folder', type=str, nargs='+', required=True,
                            help='Model folders where to load weights from.')
        parser.add_argument('-o', '--out', type=str, required=False, default=None,
                            help='Output file name. If not specified the model folder name will be used.')
        parser.add_argument('-t', '--task', type=int, nargs='+', default=-1,
                            help='Evaluate the model loading weights related to selected tasks. '
                                 'Use -1 (default) to load the last task.')
        parser.add_argument('-e', '--epoch', type=int,  nargs='+', default=-1,
                            help='Evaluate model loading weights at the selected epochs of the selected task. '
                                 'Use -1 (default) to load weights at best validation epoch.')
        parser.add_argument('--ratt-bin-forward', action='store_true', default=False,
                            help="Force binarization of forward masks for RATT approach")


    else: # TRAIN
        parser.add_argument('-f', '--folder', type=str, required=True,
                            help='Model folder where to load/save weights and csv files.')
        parser.add_argument('-a', '--approach', type=str, default='ft',
                            choices=['ft', 'ewc', 'lwf', 'ratt', 'ratt_ablation'])
        parser.add_argument('-t', '--task', type=int, default=0,
                            help='Continue training the model from the selected task.')
        parser.add_argument('-e', '--epoch', type=int, default=1,
                            help='Continue training the model from the selected epoch, loading weights from previous one.'
                                 'If epoch 1 is chosen, best epoch of the previous task will be loaded.')
        parser.add_argument('-l', '--load', type=str, default=None,
                            help='Load the best model weights from the first task of selected model/folder')

        # Model Params
        parser.add_argument('--hidden-size', type=int, default=512,
                            help="Number of neurons in LSTM hidden layer (hidden-state size)")
        parser.add_argument('--emb-size', type=int, default=256,
                            help='Number of neurons in image and word embedding layers (LSTM input size)')
        parser.add_argument('-mdl', '--max-decode-len', type=int, default=20,
                            help='Max decoding lenght for sampling (evaluation)')

        # Approach Params
        parser.add_argument('-ne', '--nb-epochs', '--nepochs', type=int, default=20,
                            help="Number of training epoch to run on each task.")
        parser.add_argument('-ex', '--examples', type=int, default=None,
                            help='Number of examples to use in each task during training, useful to speedup debugging.')
        parser.add_argument('-ee', '--extra-epochs', type=int, default=0, help='Extra epochs for the first task.')
        parser.add_argument('-lr', type=float, default=4e-4,
                            help='Learning rate for Adam optimization algorithm')
        parser.add_argument('-wd', type=float, default=0.0,
                            help='Weight decay regularization.')
        parser.add_argument('--freeze-old-words', action='store_true', default=False,
                            help='Prevent words to be trained in current task when they appeared in one of the previous tasks')

        # EWC Parameters
        parser.add_argument('--ewc-sampling', type=str, default='true', choices=['true', 'max_pred', 'multinomial'])
        parser.add_argument('--ewc-teacher-forcing', action='store_true', default=False,
                            help='Enable teacher forcing when computing fisher matrix')
        parser.add_argument('--ewc-lambda', type=float, default=1.0, help='Loss multiplier applied to EWC loss')

        # LWF Parameters
        parser.add_argument('--lwf-lambda', type=float, default=1.0,  help='Loss multiplier applied to LwF loss')
        parser.add_argument('--lwf-T', type=float, default=1.0, help='Temperature for LwF loss.')
        parser.add_argument('--lwf-h-distill', action='store_true', default=False,
                            help='Distill hidden state together with output predictions.')
        parser.add_argument('--lwf-h-lambda', type=float, default=1.0,
                            help='Loss multiplier applied to hidden state LwF loss, when --lwf-h-distill is enabled')

        # HAT Parameters
        parser.add_argument('--ratt-lambda', type=int, default=5000, help='Loss multiplier applied to RATT loss')
        parser.add_argument('--ratt-thres-cosh', type=int, default=50)
        parser.add_argument('--ratt-smax', type=int, default=400, help='Maximum value for scaling parameter s.')
        parser.add_argument('--ratt-usage', type=float, default=50, help='Network usage at the beginning of the train task.')
        parser.add_argument('--ratt-bin-backward', action='store_true', default=False,
                            help='Binarize RATT bacwkard masks.')
        parser.add_argument('--ratt-bin-forward', action='store_true', default=False,
                            help='Binarize RATT forwward masks.')

        # EXTRA HAT Ablation parameters
        parser.add_argument('--ratt-emb', action='store_true', default=False,
                            help='Enable masks for Embedding layers when executing RATT ablation')
        parser.add_argument('--ratt-cls', action='store_true', default=False,
                            help='Enable masks for classifier layers when executing RATT ablation')


    parser.add_argument('-s', '--seed', type=int, default=42, help='Chose the seed for current experiment')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU to be used from CUDA')
    parser.add_argument('--threads', type=int, default=3, help='Number of threads that torch will be able to use.')
    parser.add_argument('--pin', action='store_true', default=False,help='Pin GPU memory.')

    args = parser.parse_args()
    return args

def init_torch(args):
    from CNIC.utils import set_seed, get_device
    import torch
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = get_device(args.gpu)
    return device

def init_job(args, train_transform, eval_transform, nb_train_examples=None, nb_val_examples=None):
    #### Set proper context for training
    dataset = args.job.split('-')[0]
    if dataset == 'coco':
        from coco_settings import dataset_dir, h5_train_path, h5_val_path, h5_test_path,\
            train_imdir, val_imdir, test_imdir
        from datasets.coco.coco_dataset import CocoDataset as Dataset
    elif dataset == 'flickr30k':
        from flickr30k_settings import dataset_dir, h5_train_path, h5_val_path, h5_test_path, \
            train_imdir, val_imdir, test_imdir
        from datasets.flickr30.flickr30k_dataset import Flickr30kDataset as Dataset
    else:
        raise RuntimeError(f'There is no settings/dataset for: {dataset}')

    vocab_path = f'{dataset_dir}/vocab.pkl'

    #### Loading Dataset and DataLoaders
    job_factory = jobs_factories[args.job]
    job = job_factory.get_job()
    Dataset.init(dataset_dir)

    h5_train = h5py.File(h5_train_path, mode='r') if h5_train_path is not None else None

    # with h5py we can use max 1 worker
    tr_workers = 1 if args.workers > 1 and h5_train is not None else args.workers
    if DEBUGGING:
        tr_workers = 0  # easier to debug without threads

    for t, task in enumerate(job.tasks):
        train_ids = list(task.train_examples_ids)[:nb_train_examples]
        trainset = Dataset(train_imdir, task.vocab, h5_train, train_ids, train_transform, train=True)
        train_loader = trainset.get_loader(args.bs, shuffle=True, num_workers=tr_workers, pin_memory=args.pin,
                                           drop_last=False)
        task.train_loader = train_loader

        if args.test:
            eval_ids = list(task.test_examples_ids)[:nb_val_examples]
            h5_eval = h5py.File(h5_test_path, mode='r') if h5_val_path is not None else None
            eval_imdir = test_imdir

        else:
            h5_eval = h5py.File(h5_val_path, mode='r') if h5_val_path is not None else None
            eval_ids = list(task.val_examples_ids)[:nb_val_examples]
            eval_imdir = val_imdir

        evalset = Dataset(eval_imdir, task.vocab, h5_eval, eval_ids, eval_transform, multicaption=True, train=False)

        val_workers = 1 if args.workers > 1 and h5_eval is not None else args.workers
        if DEBUGGING:
            val_workers = 0  # easier to debug without threads
        val_loader = evalset.get_loader(args.bs, shuffle=False, num_workers=val_workers, pin_memory=args.pin,
                                       drop_last=False)
        task.val_loader = val_loader

    return job


def init_approach(args, job, cnn_feats_size, device):

    #### Building Model
    dargs = args.__dict__
    # TODO: only for retro-compatibility with older models
    if not hasattr(args, 'wd'):
        args.wd = 0
    if not hasattr(args, 'nb_epochs'):
        args.nb_epochs = args.nepochs if hasattr(args, 'nepochs') else 20
    if not hasattr(args, 'freeze_old_words'):
        args.freeze_old_words = False
    if not hasattr(args, 'embed_size'):
        args.embed_size = 256
    if not hasattr(args, 'hidden_size'):
        args.hidden_size = 512

    encoder_cnn = EncoderCNN()
    decoder = DecoderLSTMCell(cnn_feats_size, args.emb_size, args.hidden_size, job.full_vocab, args.max_decode_len)


    #### Creating Approach
    if args.approach == 'ft':
        # settings = Approach.Settings(dargs, lr, nb_epochs, freeze_old_words=False)
        settings = Approach.Settings(dargs, args.lr, args.wd, args.nb_epochs, args.extra_epochs, args.freeze_old_words)
        appr = Approach(encoder_cnn, decoder, settings, device)

    elif args.approach == 'ewc':
        # settings = ApproachEWC.Settings(dargs, lr, nb_epochs, extra_epochs, freeze_old_words=False, ewc_sampling_type='multinomial', ewc_teacher_forcing=False, ewc_lambda=100.0)
        # settings = ApproachEWC.Settings(dargs, lr, nb_epochs, extra_epochs, freeze_old_words=False, ewc_sampling_type='max_pred', ewc_teacher_forcing=False, ewc_lambda=1.0)
        settings = ApproachEWC.Settings(dargs, args.lr, args.wd, args.nb_epochs, args.extra_epochs, args.freeze_old_words,
                                        ewc_sampling_type=args.ewc_sampling, ewc_teacher_forcing=args.ewc_teacher_forcing, ewc_lambda=args.ewc_lambda)
        appr = ApproachEWC(encoder_cnn, decoder, settings, device)

    elif args.approach == 'lwf':
        # settings = ApproachLwF_3.Settings(dargs, lr, nb_epochs, lwf_lambda=1.0, lwf_T=1.0, lwf_h_distillation=True, lwf_h_lambda=1.0)
        settings = ApproachLwF_3.Settings(dargs, args.lr, args.wd, args.nb_epochs, args.extra_epochs, args.freeze_old_words,
                                          lwf_lambda=args.lwf_lambda, lwf_T=args.lwf_T,
                                          lwf_h_distillation=args.lwf_h_distill, lwf_h_lambda=args.lwf_h_lambda)
        appr = ApproachLwF_3(encoder_cnn, decoder, settings, device)


    elif args.approach == 'ratt':
        args.ratt_usage = int(args.ratt_usage * 100) if (isinstance(args.ratt_usage, float) and args.ratt_usage <= 1.) else int(args.ratt_usage)
        settings = ApproachRATT.Settings(dargs, args.lr, args.wd, args.nb_epochs, args.extra_epochs, args.freeze_old_words,
                                         lambd=args.ratt_lambda, smax=args.ratt_smax, thres_cosh=args.ratt_thres_cosh,
                                         usage=args.ratt_usage,
                                         binary_backward_masks=args.ratt_bin_backward,
                                         binary_sample_forward_masks=args.ratt_bin_forward)
        decoder = DecoderLSTMCellRATT(cnn_feats_size, args.embed_size, args.hidden_size, job.full_vocab, len(job.tasks),
                                      args.ratt_smax, args.max_decode_len, args.ratt_usage / 100.)
        appr = ApproachRATT(encoder_cnn, decoder, settings, device)

    elif args.approach == 'ratt_ablation':
        args.ratt_usage = int(args.ratt_usage * 100) if (
                    isinstance(args.ratt_usage, float) and args.ratt_usage <= 1.) else int(args.ratt_usage)
        settings = ApproachRATTAblation.Settings(dargs, args.lr, args.wd, args.nb_epochs, args.extra_epochs,
                                                 args.freeze_old_words,
                                                 lambd=args.ratt_lambda, smax=args.ratt_smax,
                                                 thres_cosh=args.ratt_thres_cosh,
                                                 usage=args.ratt_usage,
                                                 binary_backward_masks=args.ratt_bin_backward,
                                                 binary_sample_forward_masks=args.ratt_bin_forward,
                                                 ratt_cls=args.ratt_cls, ratt_emb=args.ratt_emb)
        decoder = DecoderLSTMCellRATTAblation(cnn_feats_size, args.embed_size, args.hidden_size, job.full_vocab, len(job.tasks),
                                              args.ratt_smax, args.max_decode_len, args.ratt_usage / 100.)
        appr = ApproachRATTAblation(encoder_cnn, decoder, settings, device)

    else:
        raise ValueError()

    return appr