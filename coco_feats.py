import argparse

if __name__ == '__main__':
    from coco_settings import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', help='Process validation set')
    parser.add_argument('--image_resize', type=int, default=resize_size, help='size for image after processing')
    parser.add_argument('--resized-path', action='store_true', default=False, help='use already resized images')
    parser.add_argument('--image_crop', type=int, default=224, help='size for image after processing')
    parser.add_argument('-bs', '--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--pin', action='store_true', default=False)
    args = parser.parse_args()
    args.image_dir = val_imdir if args.val else train_imdir



    import torch
    from torchvision.transforms import transforms
    from tqdm import tqdm

    from CNIC.model import EncoderCNN
    from CNIC.utils import get_device, set_seed
    from datasets.coco.coco_dataset import CocoDataset
    import numpy as np
    import h5py
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = get_device(args.gpu)


    Ts = []

    if args.resized_path:
        args.image_resize = None
        dir = resized_val_imdir if args.val else resized_train_imdir
    else:
        dir = val_imdir if args.val else train_imdir

    if args.image_resize is not None:
        Ts.append(transforms.Resize(args.image_resize))
    if args.image_crop is not None:
        Ts.append(transforms.CenterCrop(args.image_crop))
    Ts.append(transforms.ToTensor())
    Ts.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transform = transforms.Compose(Ts)

    CocoDataset.init(dataset_dir)
    cocodataset = CocoDataset(dir, transform=transform, train=not args.val, use_extra=True)
    data_loader = cocodataset.get_loader(args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.pin, drop_last=False)
    encoder_cnn = EncoderCNN().to(device)

    all_feats = []
    filenames = []
    img_ids = []
    h5_path = h5_val_path if args.val else h5_train_path
    for img, extra in tqdm(data_loader):
        with torch.no_grad():
            feats = encoder_cnn(img.to(device))
        all_feats.append(feats.detach().cpu())
        filenames += [e['file_name'].encode("ascii") for e in extra]
        img_ids += [e['image_id'] for e in extra]

    with h5py.File(h5_path, 'w') as h5f:
        h5f['resnet152'] = torch.cat(all_feats, dim=0).numpy()
        h5f['filenames'] = np.array(filenames)
        h5f['image_ids'] = np.array(img_ids)

