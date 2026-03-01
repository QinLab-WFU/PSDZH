import configparser
import os.path as osp
import pickle
import platform
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T


def get_class_num(name):
    r = {"awa2": 50, "cub": 200, "sun": 717}[name]  # seen+unseen
    return r


def get_topk(name):
    r = {"awa2": 1000, "cub": 1000, "sun": 1000}[name]
    return r


def get_concepts(name, root):
    with open(osp.join(root, name, "concepts.txt"), "r") as f:
        lines = f.read().splitlines()
    return np.array(lines)


def get_atts(name, root, normalize=True):
    """class semantic vectors from dataset"""
    atts = np.load(osp.join(root, name, "att_list.npy"))
    atts = torch.from_numpy(atts)
    """
        if name == 'cub':
            assert atts.shape == (200, 312)
        elif name == 'sun':
            assert atts.shape == (717, 102)
        elif name == 'awa2':
            assert atts.shape == (50, 85)
        """
    if normalize:
        atts = F.normalize(atts)
    return atts


def get_w2vs(name, root, normalize=True):
    """semantic attribute vectors
    learned by a language model (i.e., GloVe)"""
    # word2vec_300d
    # data from: https://github.com/hbdat/cvpr20_DAZLE
    with open(osp.join(root, name, "w2v_list.pkl"), "rb") as f:
        w2vs = pickle.load(f)
    w2vs = torch.from_numpy(w2vs.astype(np.float32))
    """
    if name == 'cub':
        assert w2vs.shape == (312, 300)
    elif name == 'sun':
        assert w2vs.shape == (102, 300)
    elif name == 'awa2':
        assert w2vs.shape == (85, 300)
    """
    if normalize:
        w2vs = F.normalize(w2vs)
    return w2vs  # shape is n_atts x 300


def build_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        steps = [T.RandomCrop(crop_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.CenterCrop(crop_size)]
    return T.Compose(
        [T.Resize(resize_size)]
        + steps
        + [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loaders(name, root, **kwargs):
    train_trans = build_trans("train")
    other_trans = build_trans("other")

    samples_per_class = kwargs.pop("samples_per_class", None)

    data = init_dataset(name, root)
    # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    query_loader = DataLoader(ImageDataset(data.query, other_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, other_trans), generator=torch.Generator(), **kwargs)

    # special process for train
    if samples_per_class is None:
        train_loader = DataLoader(ImageDataset(data.train, train_trans), shuffle=True, drop_last=True, **kwargs)
    else:
        train_set = ImageDataset(data.train, train_trans)
        batch_size = kwargs.pop("batch_size")  # must below code of query & dbase loader
        sampler = BalancedSampler(train_set, batch_size=batch_size, samples_per_class=samples_per_class)
        train_loader = DataLoader(train_set, batch_sampler=sampler, **kwargs)

    return train_loader, query_loader, dbase_loader


class BaseDataset(object):
    """
    Base class of dataset
    """

    def __init__(self, name, idx_root, img_root, verbose=True):

        self.name = name
        self.img_root = img_root

        self.img_list = osp.join(idx_root, "img_list.txt")
        self.lab_list = osp.join(idx_root, "lab_list.txt")
        self.idx_list = osp.join(idx_root, "idx_list.pkl")

        self.check_before_run()

        self.train, self.query, self.dbase = self.process()

        self.set_img_abspath()  # 1.jpg -> /home/x/SUN/images/1.jpg

        if verbose:
            print(f"=> {name.upper()} loaded")
            self.print_dataset_statistics()

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.img_list):
            raise RuntimeError("'{}' is not available".format(self.img_list))
        if not osp.exists(self.lab_list):
            raise RuntimeError("'{}' is not available".format(self.lab_list))
        if not osp.exists(self.idx_list):
            raise RuntimeError("'{}' is not available".format(self.idx_list))

    def get_imagedata_info(self, data):
        labs = data[1]
        n_cids = (labs.sum(axis=0) > 0).sum()  # len(np.unique(labs))
        n_imgs = len(data[0])
        return n_cids, n_imgs

    def print_dataset_statistics(self):
        n_train_cids, n_train_imgs = self.get_imagedata_info(self.train)
        n_query_cids, n_query_imgs = self.get_imagedata_info(self.query)
        n_dbase_cids, n_dbase_imgs = self.get_imagedata_info(self.dbase)

        print("Image Dataset statistics:")
        print("  -----------------------------")
        print("  subset | # images | # classes")
        print("  -----------------------------")
        print("  train  | {:8d} | {:9d}".format(n_train_imgs, n_train_cids))
        print("  query  | {:8d} | {:9d}".format(n_query_imgs, n_query_cids))
        print("  dbase  | {:8d} | {:9d}".format(n_dbase_imgs, n_dbase_cids))
        print("  -----------------------------")

    def process(self):
        labs = np.loadtxt(self.lab_list, dtype=int)
        # single -> onehot
        n_classes = get_class_num(self.name)
        labs = np.eye(n_classes, dtype=np.float32)[labs - 1]

        imgs = np.loadtxt(self.img_list, dtype=str)

        with open(self.idx_list, "rb") as f:
            data = pickle.load(f)

        train_idxes = data["trainval_loc"]
        query_idxes = data["test_unseen_loc"]
        dbase_idxes = np.concatenate((data["test_seen_loc"], query_idxes), axis=0)

        return (
            (imgs[train_idxes], labs[train_idxes]),
            (imgs[query_idxes], labs[query_idxes]),
            (imgs[dbase_idxes], labs[dbase_idxes]),
        )

    def set_img_abspath(self):
        for x in ["train", "query", "dbase"]:
            imgs, labs = getattr(self, x)
            # imgs = [osp.join(self.img_root, img) for img in imgs]
            imgs = np.char.add(f"{self.img_root}/", imgs)
            setattr(self, x, (imgs, labs))


def init_dataset(name, root, **kwargs):
    idx_root = osp.join(root, name)

    ini_loc = osp.join(root, name, "images", "location.ini")
    if osp.exists(ini_loc):
        config = configparser.ConfigParser()
        config.read(ini_loc)
        if "wfu.edu.cn" in platform.node():
            img_root = config["DEFAULT"]["SLURM"]
        else:
            img_root = config["DEFAULT"][platform.system()]
    else:
        img_root = osp.join(root, name)

    return BaseDataset(name, idx_root, img_root, **kwargs)


class ImageDataset(Dataset):
    """Image Dataset"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        img, lab = self.data[0][idx], self.data[1][idx]

        # img path -> img tensor
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, lab, idx

    def get_all_labels(self):
        return torch.from_numpy(self.data[1])


class BalancedSampler(Sampler):
    """
    BalancedSampler ensures each batch contains an equal number of samples
    from randomly selected classes. This is especially useful for
    imbalanced datasets during training.
    """

    def __init__(self, dataset: Dataset, batch_size, samples_per_class):
        super().__init__()
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.all_labels = dataset.get_all_labels()
        self.classes = (self.all_labels.sum(dim=0) > 0).nonzero(as_tuple=True)[0].tolist()
        self.cls_dict = {c: (self.all_labels[:, c] == 1).nonzero(as_tuple=True)[0].tolist() for c in self.classes}

        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.sampler_length = self.n_samples // batch_size

        assert (
            batch_size % samples_per_class == 0
        ), f"batch_size ({batch_size}) must be divisible by samples_per_class ({samples_per_class})!"

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            for _ in range(self.batch_size // self.samples_per_class):
                class_key = random.choice(self.classes)

                # only calc class_sample_idxes once!
                x = self.cls_dict[class_key]
                # x = (self.all_labels[:, class_key] == 1).nonzero(as_tuple=True)[0].tolist()

                class_idx_list = random.sample(x, k=self.samples_per_class)
                subset.extend(class_idx_list)

            yield subset

    def __len__(self):
        return self.sampler_length


def _test_sampler():
    from _data_zs import init_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    ds = init_dataset("cub", "../_datasets_zs")

    trans = T.Compose(
        [
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(ds.train, trans)
    sampler = BalancedSampler(train_set, batch_size=90, samples_per_class=2)
    print("len(sampler)", len(sampler))

    dataloader = DataLoader(train_set, batch_sampler=sampler)

    for images, labels, _ in dataloader:
        print(images.shape, labels.shape)
        print(labels.argmax(dim=1))
        break


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    db_name = "sun"
    root = "./_datasets_zs"

    dataset = init_dataset(db_name, root)

    trans = T.Compose(
        [
            # T.ToPILImage(),
            T.Resize([224, 224]),
            T.ToTensor(),
        ]
    )

    train_set = ImageDataset(dataset.train, trans)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)
    seen_idxes = dataloader.dataset.get_all_labels().sum(dim=0).nonzero(as_tuple=True)[0]

    concepts = get_concepts(db_name, root)

    atts = get_atts(db_name, root).to("cuda:1")
    print((atts < 0).nonzero())
    exit()
    print("atts.shape", atts.shape)

    w2vs = get_w2vs(db_name, root)
    print("w2vs.shape", w2vs.shape)
    exit()

    for imgs, labs, _ in dataloader:
        print(imgs.shape, labs[0].nonzero())
        plt.imshow(imgs[0].numpy().transpose(1, 2, 0))
        plt.title(concepts[labs[0].argmax()])
        plt.show()
        break
