import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import argparse


dataset_attributes = {
    'CelebA': {
        'root_dir': 'celebA'
    },
    'CUB': {
        'root_dir': 'cub'
    },
    'CIFAR10': {
        'root_dir': 'CIFAR10/data'
    },
    'MultiNLI': {
        'root_dir': 'multinli'
    }
}

shift_types = ['confounder', 'label_shift_step']

model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    }
}


from torch.utils.data import Dataset, Subset
class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        if model_attributes[self.model_type]['feature_type']=='precomputed':
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
                img = self.eval_transform(img)
            # Flatten if needed
            if model_attributes[self.model_type]['flatten']:
                assert img.dim()==3
                img = img.view(-1)
            x = img

        return x,y,g

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
    

class CUBDataset(ConfounderDataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        self.data_dir = os.path.join(
            self.root_dir,
            'data',
            '_'.join([self.target_name] + self.confounder_names))

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Set transform
        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_cub(
                self.model_type,
                train=True,
                augment_data=augment_data)
            self.eval_transform = get_transform_cub(
                self.model_type,
                train=False,
                augment_data=augment_data)


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0/224.0
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform

confounder_settings = {
    'CUB':{
        'constructor': CUBDataset
    }

}


import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []

        for x,y,g in self:
            group_array.append(g)
            y_array.append(y)
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()

    def get_loader(self, args, train, reweight_groups, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset,
                num_replicas=args.ngpu,
                rank=args.local_rank,
                )
        loader = DataLoader(self.dataset,batch_size=args.batch_size, num_workers=4,
                    pin_memory=False,
                    shuffle=True,
                    sampler=sampler,
                )
        '''
            loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        '''

        return loader

def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type=='confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)

def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')


def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]['constructor'](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data)
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str)
    if train:
        splits = ['train', 'val', 'test']
    else:
        splits = ['test']
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets


def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio
        
parser = argparse.ArgumentParser()

# Settings
parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), default = 'CUB', required=True)
parser.add_argument('-s', '--shift_type', choices=shift_types, default='confounder',required=True)
# Confounders
parser.add_argument('-t', '--target_name',default='waterbird_complete95')
parser.add_argument('-c', '--confounder_names', default = 'forest2water2', nargs='+')
# Resume?
parser.add_argument('--resume', default=False, action='store_true')
# Label shifts
parser.add_argument('--minority_fraction', type=float)
parser.add_argument('--imbalance_ratio', type=float)
# Data
parser.add_argument('--fraction', type=float, default=1.0)
parser.add_argument('--root_dir', default='data/cub')
parser.add_argument('--reweight_groups', action='store_true', default=False)
parser.add_argument('--augment_data', action='store_true', default=False)
parser.add_argument('--val_fraction', type=float, default=0.1)
# Objective
parser.add_argument('--robust', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--generalization_adjustment', default="0.0")
parser.add_argument('--automatic_adjustment', default=False, action='store_true')
parser.add_argument('--robust_step_size', default=0.01, type=float)
parser.add_argument('--use_normalized_loss', default=False, action='store_true')
parser.add_argument('--btl', default=False, action='store_true')
parser.add_argument('--hinge', default=False, action='store_true')

# Model
parser.add_argument(
    '--model',
    choices=model_attributes.keys(),
    default='resnet50')
parser.add_argument('--train_from_scratch', action='store_true', default=False)

# Optimization
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--scheduler', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--minimum_variational_weight', type=float, default=0)
# Misc
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--show_progress', default=False, action='store_true')
parser.add_argument('--log_dir', default='./logs')
parser.add_argument('--log_every', default=50, type=int)
parser.add_argument('--save_step', type=int, default=10)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--save_last', action='store_true', default=False)
parser.add_argument('--local-rank', type =int, default = 0)
parser.add_argument('--ngpu', type=int, default=2)

args = parser.parse_args()
check_args(args)
ngpus_per_node = torch.cuda.device_count()

import pickle
with open('data/save/cub_train.pkl', 'rb') as fp:
    train_data = pickle.load(fp) 


def checkpoint(model, acc, epoch, optimizer, save_name_add=''):
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'acc': acc,
        'model': model.state_dict(),
        'optimizer_state' : optimizer.state_dict(),
        'rng_state': torch.get_rng_state()
    }

    save_name = './checkpoint/ckpt.t7' + '_'
    save_name += save_name_add

    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    torch.save(state, save_name)

ngpus_per_node = torch.cuda.device_count()

def train(train_loader, model, Rep, optimizer, criterion):
    model.train()
    total_loss = 0.0
    print_freq = 200
    for idx, (input, group, target) in enumerate(train_loader):
        # print(input[0,1,1,:])
        input = input.cuda()
        group = group.cuda()

        #import pdb; pdb.set_trace()
        advinputs, adv_loss = Rep.get_loss(original_images=input,target=input, optimizer=optimizer, weight=1.0, labels=group, criterion=criterion, random_start=True)
        loss = adv_loss
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        optimizer.step()
        
        if idx % print_freq:
            info = "Loss at Batch " + idx + ": " + total_loss/(idx+1)
            print(info)
    return total_loss/idx

def check(model, projector, epoch, train_loss, optimizer):
    model.eval()
    projector.eval()

    # Save at the last epoch #       
    if epoch == args.n_epochs - 1 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, train_loss, epoch, args, optimizer)
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add='_projector')
       
    # Save at every 100 epoch #
    elif epoch % 50 == 0 and args.local_rank % ngpus_per_node == 0:
        checkpoint(model, train_loss, epoch, args, optimizer, save_name_add='_epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add=('_projector_epoch_' + str(epoch)))


from RoCL.src.attack_lib import RepresentationAdv
from SupContrast.networks.resnet_big import SupConResNet
from SupContrast.losses import SupConLoss
from RoCL.src.models.projector import Projector
from torchlars import LARS
import torch.optim as optim
import csv
import torch.backends.cudnn as cudnn

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#local_rank = 0
print("Parallel Initializing...")
import os
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
# dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)

world_size = torch.cuda.device_count()
torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank= args.local_rank,
)
# print(local_rank)
torch.cuda.set_device(args.local_rank)

def get_loader(dataset, args):
    sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.ngpu,
            rank=args.local_rank,
            )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                pin_memory=False,
                shuffle=(sampler is None),
                sampler=sampler,
            )
    return loader


##loader_kwargs = {'batch_size':16, 'num_workers':1, 'pin_memory':True}
#train_loader = train_data.get_loader(train=True, args=args, reweight_groups=args.reweight_groups, **loader_kwargs)
train_loader = get_loader(train_data.dataset,args)
print("Data Loaded...")


# device = torch.device("cpu")
# torch.cuda.empty_cache()
print("Model Initializing...")

model = SupConResNet().cuda()
criterion = SupConLoss().cuda()
projector = Projector(expansion=4).cuda()
Rep = RepresentationAdv(model, projector, epsilon=0.0314, alpha=0.007, min_val=0.0, max_val=1.0, max_iters=7,_type='linf', loss_type='sup', regularize ='other')


print("Model Paralleling...")
model       = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model       = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
projector   = torch.nn.parallel.DistributedDataParallel(
                projector,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
)
cudnn.benchmark = True


model_params = []
model_params += model.parameters()
model_params += projector.parameters()
optimizer  = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-6)
# optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)


seed = 42
torch.manual_seed(seed)
loginfo = 'results/log_' + str(seed)
logname = (loginfo+ '.csv')

print("Start Training")

# num_epochs = 100
for epoch in range(args.n_epochs):
    train_loss = train(train_loader,model, Rep,optimizer,criterion)
    check(model, projector, epoch, train_loss, optimizer)

    if args.local_rank % ngpus_per_node == 0:
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss.item()])