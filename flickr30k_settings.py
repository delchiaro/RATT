dataset_dir: str = '/equilibrium/delchiaro/datasets/flickr30k'
#dataset_dir = '/home/btwardow/datasets/flickr30k'

h5_path_fmt = '{}/resnet152_{}.h5'
resized_imdir_fmt = '{}/resized_{}'
resize_size = 256

SPLITS = ('train', 'val', 'test')
h5_path = {split: h5_path_fmt.format(dataset_dir, split) for split in SPLITS}
h5_train_path, h5_val_path, h5_test_path = [h5_path[s] for s in SPLITS]
resized_imdir = {split: resized_imdir_fmt.format(dataset_dir, split) for split in SPLITS}
resized_train_imdir, resized_val_imdir, resized_test_imdir = [resized_imdir[s] for s in SPLITS]

# for Flickr30k splits are first arg to Dataset, which in Coco is dir
train_imdir, val_imdir, test_imdir = [s for s in SPLITS]


vocab_path = f'{dataset_dir}/vocab.pkl'
