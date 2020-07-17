from collections import OrderedDict

from datasets.coco.coco_job import CocoJobFactory
from datasets.flickr30.flickr30k_job import Flickr30kJobFactory
import flickr30k_settings
import coco_settings


######################## COCO JOBS
t_transport = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
t_animals_nohome = ['bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'] #, 'cat', 'dog']
t_animals = ['bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'cat', 'dog']
t_sports = ['snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket']
t_food = ['banana', 'apple', 'sandwich' ,'orange', 'broccoli', 'carrot','hot dog', 'pizza', 'donut', 'cake']
t_interior = ['chair','couch','potted plant','bed','toilet','tv','laptop', 'mouse','remote', 'keyboard',
              'cell phone','microwave','oven','toaster','sink', 'refrigerator', 'dining table']
t_interior_nodinning = ['chair','couch','potted plant','bed','toilet','tv','laptop', 'mouse','remote', 'keyboard',
              'cell phone','microwave','oven','toaster','sink', 'refrigerator'] #, 'dining table']

coco_TASFI = CocoJobFactory(coco_settings.dataset_dir,
                            {'transport': t_transport,
                               'animals': t_animals_nohome,
                               'sports': t_sports,
                               'food': t_food,
                               'interior': t_interior_nodinning},
                            job_name='TASFI')


######################## FLICKR30K JOBS

# SPLIT1 = ('people', 'scene', 'animals', 'vehicles', 'instruments')
# SPLIT1_MAX_SAMPLES = {'train': 5000, 'val': 170, 'test': 170}
# SPLIT1_TASKS = OrderedDict([(f'T{s}', [s]) for s in SPLIT1])
#
# flickr30k_split1 = Flickr30kJobFactory(flickr30k_settings.dataset_dir, SPLIT1_TASKS, SPLIT1_MAX_SAMPLES, job_name='PSAVI')
#
#
#
# SPLIT1 = ('people', 'scene', 'animals', 'vehicles')
# SPLIT1_MAX_SAMPLES = {'train': 5000, 'val': 170, 'test': 170}
# SPLIT1_TASKS = OrderedDict([(f'T{s}', [s]) for s in SPLIT1])
#
# flickr30k_split2 = Flickr30kJobFactory(flickr30k_settings.dataset_dir, SPLIT1_TASKS, SPLIT1_MAX_SAMPLES, job_name='PSAV')



SPLIT1 = ('scene', 'animals', 'vehicles', 'instruments')
# SPLIT1_MAX_SAMPLES = {'train': 5000, 'val': 170, 'test': 170}
SPLIT1_MAX_SAMPLES = {'train': 7000, 'val': 250, 'test': 250}
SPLIT1_TASKS = OrderedDict([(f'T{s}', [s]) for s in SPLIT1])
flickr30k_split = Flickr30kJobFactory(flickr30k_settings.dataset_dir, SPLIT1_TASKS, SPLIT1_MAX_SAMPLES, job_name='SAVI')

