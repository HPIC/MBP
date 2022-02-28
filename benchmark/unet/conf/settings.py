from datetime import datetime
#total training epoches
EPOCH = 1
MILESTONES = [40, 80, 120]

#mean and std of cifar100 dataset
CARVANA_MEAN = (0.485, 0.456, 0.406)
CARVANA_STD = (0.229, 0.224, 0.225)
# CARVANA_MEAN = (0.0, 0.0, 0.0)
# CARVANA_STD = (1.0, 1.0, 1.0)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10
