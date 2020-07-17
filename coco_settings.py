dataset_dir: str = '/equilibrium/delchiaro/datasets/mscoco14'
#dataset_dir: str = '/home/btwardow/datasets/mscoco14'

train_imdir = f'{dataset_dir}/coco_images/train2014'
val_imdir = f'{dataset_dir}/coco_images/val2014'
test_imdir = f'{dataset_dir}/coco_images/val2014'

resized_train_imdir = f'{dataset_dir}/coco_images/resized_train2014'
resized_val_imdir = f'{dataset_dir}/coco_images/resized_val2014'
resized_test_imdir = f'{dataset_dir}/coco_images/resized_val2014'
resize_size = 256

h5_train_path = f'{dataset_dir}/coco_images/resnet152_train2014.h5'
h5_val_path = f'{dataset_dir}/coco_images/resnet152_val2014.h5'
h5_test_path = f'{dataset_dir}/coco_images/resnet152_val2014.h5'

vocab_path = f'{dataset_dir}/vocab.pkl'

