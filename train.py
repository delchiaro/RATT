import torch
from torchvision.transforms import transforms
from CNIC.utils import set_seed, get_device
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
args = parse_args()
device = init_torch(args)

#### Paths
model_dir = f'models/{args.folder}'
model_load_dir = model_dir if args.load is None else f'models/{args.load}'


job = init_job(args, train_transform, val_transform, args.examples, args.examples)
appr = init_approach(args, job, cnn_feats_size, device)
appr.train_job(job, start_t=args.task, start_ep=args.epoch, log_step=50, model_dir=model_dir)

