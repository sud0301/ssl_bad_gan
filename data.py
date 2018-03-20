
import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import *

class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0), self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len

def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i][0] == 10:
                data_set.labels[i][0] = 0
    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: config.size_labeled_data / 10]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_cifar_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    dev_set = CIFAR10('cifar', train=False, download=True, transform=transform)
    
    print (type(training_set[0]))
    print (len(dev_set))    
	 
    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_gris_loaders(config):
    transform = transforms.Compose([transforms.Resize(size=(32, 32), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #training_set = CIFAR10('cifar', train=True, download=True, transform=transform)
    #dev_set = CIFAR10('cifar', train=False, download=True, transform=transform)
    
    training_set = ImageFolder('/misc/lmbraid19/mittal/iclr_16/dataset_10_GRIS_pytorch/', transform=transform)   
 
    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]

    labeled_indices, test_indices_uf = indices[mask], indices[~mask]
    print ('# Labeled indices ', len(labeled_indices) )
    print ('# Unlabeled indices All ', len(test_indices_uf) )
	
    test_mask = np.zeros(test_indices_uf.shape[0], dtype=np.bool)
    test_labels = np.array([training_set[i][1] for i in test_indices_uf], dtype=np.int64)
    for i in range(10):
        test_mask[np.where(test_labels == i)[0][: int(config.size_test_data / 10)]] = True
    test_indices, unlabeled_indices_all = test_indices_uf[test_mask], test_indices_uf[~test_mask] 
    
    unlabeled_indices = unlabeled_indices_all[:15000]
    print ('# Unlabeled indices train', len(unlabeled_indices))
    print ('# Test indices ', len(test_indices))
	
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', test_indices.shape[0])

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, training_set, test_indices, config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set


def get_pr2_loaders(config):
    #transform = transforms.Compose([transforms.Resize(size=(32, 32), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if config.image_side==256:
        transform = transforms.Compose([ transforms.Resize(size=(224, 224), interpolation=2),  transforms.Resize(size=(config.image_side, config.image_side), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([ transforms.Resize(size=(config.image_side, config.image_side), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transform = transforms.Compose([transforms.Resize(size=(config.image_side, config.image_side), interpolation=2), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225 ))])
   
    train_labeled_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/dataset_splits/20180220/train_labeled_sample/train_set_400_1/', transform=transform)   
 
    train_labeled_indices = np.arange(len(train_labeled_set))
    np.random.shuffle(train_labeled_indices)
    mask = np.zeros(train_labeled_indices.shape[0], dtype=np.bool)
    labels = np.array([train_labeled_set[i][1] for i in train_labeled_indices], dtype=np.int64)
    	
    for i in range(7):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 7)]] = True
    
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    '''
    for i in range(len(train_labeled_set)):
        if (int(train_labeled_set[i][1]) == 6):
            img_name ='image' + str(i) + '_' + str(train_labeled_set[i][1]) + '.jpg'
            vutils.save_image(train_labeled_set[i][0], img_name) 
    '''

    train_labeled_indices = train_labeled_indices[mask]
    print ('# Labeled indices ', len(train_labeled_indices) )
    #print ('# Unlabeled indices All ', len(test_indices_uf) )

    train_unlabeled_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/dataset_splits/20180220/train_unlabeled/', transform=transform)   
    train_unlabeled_indices = np.arange(len(train_unlabeled_set))
    np.random.shuffle(train_unlabeled_indices)
   
    #train_unlabeled_indices = train_unlabeled_indices_all[:1000]
    print ('# UnLabeled indices ', len(train_unlabeled_indices) )
     

    if config.include_train_labeled: 	
        images_unl, labels_unl = zip(*train_unlabeled_set)
        images_lab, labels_lab = zip(*train_labeled_set)

        images_comb = images_unl + images_lab
        labels_comb = labels_unl + labels_lab
        '''
        count_comb = len(train_labeled_indices) + len(train_unlabeled_indices)
        images_comb, labels_comb = [], []
        for idx in train_labeled_indices:
            image, label = train_labeled_set[idx]
            images_comb.append(image)
            labels_comb.append(label)
        for idx in train_unlabeled_indices:
            image, label = train_unlabeled_set[idx]
            images_comb.append(image)
            labels_comb.append(label)
        '''
        images_comb = torch.stack(images_comb, 0)
        labels_comb = torch.from_numpy(np.array(labels_comb, dtype=np.int64)).squeeze()
   
        train_unlabeled_set_comb = list(zip(images_comb, labels_comb))
        train_unlabeled_indices_comb = np.arange(len(train_unlabeled_set_comb))
        np.random.shuffle(train_unlabeled_indices_comb)
        
        #train_unlabeled_indices_comb = train_unlabeled_indices_comb_all[:10000]
        print ('# UnLabeled indices combined', len(train_unlabeled_indices_comb) )
 
    test_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/dataset_splits/20180220/test_labeled_sample/test_set_1/', transform=transform)   
    #test_set = ImageFolder('/misc/lmbraid19/mittal/yolo-9000/yolo_dataset/test_set_extras/', transform=transform)   
    test_indices = np.arange(len(test_set))
    print ('# Test indices ', len(test_indices))
	
    #print ('labeled size', train_labeled_indices.shape[0], 'unlabeled size', train_unlabeled_indices.shape[0], 'dev size', test_indices.shape[0])

    labeled_loader = DataLoader(config, train_labeled_set, train_labeled_indices, config.train_batch_size)
    if config.include_train_labeled:
        unlabeled_loader = DataLoader(config, train_unlabeled_set_comb, train_unlabeled_indices_comb, config.train_batch_size_2)
        unlabeled_loader2 = DataLoader(config, train_unlabeled_set_comb, train_unlabeled_indices_comb, config.train_batch_size_2) 
    else:
        unlabeled_loader = DataLoader(config, train_unlabeled_set, train_unlabeled_indices, config.train_batch_size_2)
        unlabeled_loader2 = DataLoader(config, train_unlabeled_set, train_unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, test_set, test_indices, config.dev_batch_size)

    special_set = []
    for i in range(7):
        special_set.append(train_labeled_set[train_labeled_indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set
