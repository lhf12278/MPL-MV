import sys
sys.path.append('../')

from .dataset import *
from .loader import *

import torchvision.transforms as transforms
from tools import *
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

class ReIDLoaders:

    def __init__(self, config):

        # resize --> flip --> pad+crop --> colorjitor(optional) --> totensor+norm --> rea (optional)
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size)]
        if config.use_colorjitor: # use colorjitor
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea: # use rea
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        # resize --> totensor --> norm
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        self.datasets = ['market_duke','market_msmt','duke_msmt','duke_market','msmt_duke','msmt_market','market', 'duke', 'msmt', ]

        # dataset
        self.market_path = config.market_path
        self.duke_path = config.duke_path
        self.msmt_path = config.msmt_path

        self.train_task = config.train_task

        self.train_dataset = config.train_task.split('_')[0]
        self.test_dataset = config.train_task.split('_')[1]
        #target
        self.t_dataset = config.train_task.split('_')[1]
        self.target_modify = config.target_modify


        # batch size
        self.p = config.p
        self.k = config.k

        # load
        self._load()


    def _load(self):

        '''init train dataset'''

        s_samples, t_samples = self._get_train_samples(self.train_task)

        self.s_train_iter = self._get_uniform_iter(s_samples, self.transform_train, self.p, self.k)
        self.t_train_iter = self._get_random_iter(t_samples, self.transform_train, self.p * self.k)

        '''init test dataset'''
        if self.test_dataset == 'market':
            self.market_query_samples, self.market_gallery_samples = self._get_test_samples('market')
            self.market_query_loader = self._get_loader(self.market_query_samples, self.transform_test, 128)
            self.market_gallery_loader = self._get_loader(self.market_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'duke':
            self.duke_query_samples, self.duke_gallery_samples = self._get_test_samples('duke')
            self.duke_query_loader = self._get_loader(self.duke_query_samples, self.transform_test, 128)
            self.duke_gallery_loader = self._get_loader(self.duke_gallery_samples, self.transform_test, 128)
        elif self.test_dataset == 'msmt':
            self.msmt_query_samples, self.msmt_gallery_samples = self._get_test_samples('msmt')
            self.msmt_query_loader = self._get_loader(self.msmt_query_samples, self.transform_test, 128)
            self.msmt_gallery_loader = self._get_loader(self.msmt_gallery_samples, self.transform_test, 128)



    def _get_train_samples(self, train_task):
        '''get train samples, support multi-dataset'''

        if train_task == 'market':
            samples = Samples4Market(self.market_path, relabel=True).train
            return samples
        elif train_task == 'duke':
            samples = Samples4Duke(self.duke_path, relabel=True).train
            return samples
        elif train_task == 'msmt':
            samples = Samples4MSMT17(self.msmt_path, relabel=True).train
            return samples
        elif train_task == 'market_duke':
            samples_market = Samples4Market(self.market_path, is_target=False,train_st=self.train_task,relabel=True).train
            samples_duke = Samples4Duke(self.duke_path,is_target=True,train_st=self.train_task, relabel=True,target_modify=self.target_modify).train
            return samples_market, samples_duke
        elif train_task == 'duke_market':

            samples_duke = Samples4Duke(self.duke_path, is_target=False,train_st=self.train_task, relabel=True).train
            samples_market = Samples4Market(self.market_path, is_target=True,train_st=self.train_task, relabel=True,target_modify=self.target_modify).train
            return  samples_duke,samples_market
        elif train_task == 'market_msmt':
            samples_market = Samples4Market(self.market_path, is_target=False,train_st=self.train_task, relabel=True).train
            samples_msmt = Samples4MSMT17(self.msmt_path, is_target=True,train_st=self.train_task, relabel=True,target_modify=self.target_modify).train
            return samples_market, samples_msmt
        elif train_task == 'msmt_market':
            samples_market = Samples4Market(self.market_path, is_target=True, train_st=self.train_task,relabel=True).train
            samples_msmt = Samples4MSMT17(self.msmt_path, is_target=False, train_st=self.train_task,relabel=True).train
            return samples_msmt,samples_market
        elif train_task == 'duke_msmt':
            samples_duke = Samples4Duke(self.duke_path, is_target=False,train_st=self.train_task, relabel=True).train
            samples_msmt = Samples4MSMT17(self.msmt_path, is_target=True,train_st=self.train_task, relabel=True,target_modify=self.target_modify).train
            return samples_duke, samples_msmt
        elif train_task == 'msmt_duke':
            samples_duke = Samples4Duke(self.duke_path, is_target=True,train_st=self.train_task, relabel=True).train
            samples_msmt = Samples4MSMT17(self.msmt_path, is_target=False,train_st=self.train_task, relabel=True).train
            return  samples_msmt,samples_duke


    def _get_test_samples(self, test_dataset):
        print('Test data statistics')
        if test_dataset == 'market':
            market = Samples4Market(self.market_path, relabel=True, )
            query_samples = market.query
            gallery_samples = market.gallery
        elif test_dataset == 'duke':
            duke = Samples4Duke(self.duke_path, relabel=True, )
            query_samples = duke.query
            gallery_samples = duke.gallery
        elif 'msmt' in test_dataset:
            msmt = Samples4MSMT17(self.msmt_path,target_modify=self.target_modify)
            query_samples = msmt.query
            gallery_samples = msmt.gallery
        return query_samples, gallery_samples

    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        '''
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=True, sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)
        return iters

    def _get_random_iter(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=False)
        return loader
