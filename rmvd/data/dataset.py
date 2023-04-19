import os.path as osp
import random
import abc
import pickle
import time

import torch
import numpy as np
import pytoml

import rmvd.utils as utils
from .transforms import Resize
from .updates import Updates, PickledUpdates
from .layout import Layout


class Sample(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, root):
        return


class Dataset(torch.utils.data.Dataset, metaclass=abc.ABCMeta):
    def __init__(
        self,
        root,
        aug_fcts=None,
        input_size=None,
        to_torch=False,
        updates=None,
        update_strict=False,
        layouts=None,
        verbose=True,
        **kwargs,
    ):

        aug_fcts = [] if aug_fcts is None else aug_fcts
        aug_fcts = [aug_fcts] if not isinstance(aug_fcts, list) else aug_fcts
        self.verbose = verbose

        self.root = None
        self._init_root(root)

        if self.verbose:
            print(f"Initializing dataset {self.name} from {self.root}")

        self._seed_initialized = False
        self.resize = (
            Resize(size=input_size)
            if (
                input_size is not None
                and input_size[0] is not None
                and input_size[1] is not None
            )
            else None
        )  # TODO: handle case where input_size is scaler; move this to input_size setter
        self.aug_fcts = []
        self._init_aug_fcts(aug_fcts)
        self.to_torch = to_torch

        self.samples = []
        self._init_samples(**kwargs)
        self._layouts = {}
        self._init_layouts(layouts)
        self._allowed_indices = []
        self.updates = []
        self._init_updates(updates, update_strict)

        if self.verbose:
            print(f"\tNumber of samples: {len(self)}")
            if len(self.updates) > 0:
                print(
                    f"\tUpdates: {', '.join([update.name for update in self.updates])}"
                )
            if self.resize is not None:
                print(
                    f"\tImage resolution (height, width): ({input_size[0]}, {input_size[1]})"
                )
            print(f"Finished initializing dataset {self.name}.")
            print()

    @property
    def name(self):
        """The name property returns the name of the dataset, which is composed of the base_dataset attribute (if present), the split attribute (if present), and the dataset_type attribute (if present) separated by periods. If the base_dataset attribute is not present, the name is the name of the class."""
        if hasattr(self, "base_dataset"):
            name = self.base_dataset
            name = f"{name}.{self.split}" if hasattr(self, "split") else name
            name = (
                f"{name}.{self.dataset_type}" if hasattr(self, "dataset_type") else name
            )
        else:
            name = type(self).__name__  # class name
        return name

    @property
    def full_name(self):
        """The full_name property returns the full name of the dataset, which includes the name of the dataset and the names of any updates applied to it, separated by a + symbol"""
        name = self.name
        for update in self.updates:
            name += f"+{update.name}"
        return name

    def _init_root(self, root):
        """The _init_root method sets the root attribute of the dataset to a string or the first element in a list of strings that corresponds to an existing directory."""
        if isinstance(root, str):
            self.root = root
        elif isinstance(root, list):
            self.root = [path for path in root if osp.isdir(path)][0]

    def _init_aug_fcts(self, aug_fcts):
        """This method initializes the list of augmentation functions that will be applied to each sample during training or inference. If an element of the aug_fcts list is a string, it is assumed to be the name of an augmentation function, which is then loaded dynamically using the utils.get_function method. The resulting function object is appended to the aug_fcts list. If an element of the aug_fcts list is already a function object, it is simply appended to the list."""
        for aug_fct in aug_fcts:
            if isinstance(aug_fct, str):
                aug_fct_class = utils.get_function(aug_fct)
                aug_fct = aug_fct_class()
            self.aug_fcts.append(aug_fct)

    def _init_samples(self):
        self._init_samples_from_list()

    def _init_samples_from_list(self):
        sample_list_path = _get_sample_list_path(self.name)
        if self.verbose:
            print("\tInitializing samples from list at {}.".format(sample_list_path))
        with open(sample_list_path, "rb") as sample_list:
            self.samples = pickle.load(sample_list)

    def _init_updates(self, updates, update_strict=False):
        """The _init_updates() method initializes the dataset's updates. It takes two arguments, updates and update_strict. If updates is not None, the method will loop over each update in updates. If the update is an instance of Updates, it will be added to the dataset's updates. Otherwise, if the update is a string, it will be assumed to be a path to a pickled update and loaded using the PickledUpdates class. The loaded update is then added to the dataset's updates.

        If update_strict is True, the method will set the dataset's allowed indices to only those indices that appear in all updates. Otherwise, all indices will be allowed."""
        if updates is not None:
            for update in updates:
                if isinstance(update, Updates):
                    update = update
                elif isinstance(update, str):
                    update = PickledUpdates(path=update, verbose=False)
                self.updates.append(update)

        if update_strict:
            self._allowed_indices = [
                i
                for i in range(len(self.samples))
                if all([i in update for update in self.updates])
            ]
        else:
            self._allowed_indices = list(range(len(self.samples)))

    def _init_layouts(self, layouts):
        if layouts is not None:
            for layout in layouts:
                layout = (
                    layout if isinstance(layout, Layout) else Layout.from_file(layout)
                )
                self.add_layout(layout)

    def add_layout(self, layout):
        self._layouts[layout.name.lower()] = layout

    def get_layout_names(self):
        return list(self._layouts.keys())

    def get_layout(self, layout_name=None):
        layout_name = layout_name if layout_name is not None else "default"
        return self._layouts[layout_name.lower()]

    def __len__(self):
        return len(self._allowed_indices)

    def __getitem__(self, index):
        """The __getitem__() method returns the sample at the specified index.
        First, it gets the sample at the corresponding index in the samples list.
        Then, it initializes the seed for the current worker (if it has not already been initialized) to ensure reproducibility.
        It loads the sample's data and adds metadata to the dictionary. It then preprocesses the sample using _preprocess_sample().
        It applies each update in the dataset's updates, then applies each augmentation function in the dataset's augmentation functions.
        If resize is not None, it resizes the sample using the resize method.
        Finally, if to_torch is True, it coallates the samples into a torch.Tensor using the utils.torch_collate() method."""
        index = self._allowed_indices[index]
        sample = self.samples[index]

        if not self._seed_initialized:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self._seed_initialized = True

        sample_dict = sample.load(root=self.root)
        sample_dict["_index"] = index
        sample_dict["_dataset"] = self.full_name

        _preprocess_sample(sample_dict)

        for update in self.updates:
            update.apply_update(sample_dict, index=index)

        for aug_fct in self.aug_fcts:
            aug_fct(sample_dict)

        if self.resize is not None:
            self.resize(sample_dict)

        if self.to_torch:
            sample_dict = utils.torch_collate([sample_dict])

        return sample_dict

    def __str__(self):
        return self.name

    def _get_paths(self):
        return _get_paths()

    def _get_path(self, *keys):
        return _get_path(*keys)

    @classmethod
    def init_as_loader(
        cls,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
        indices=None,
        **kwargs,
    ):
        kwargs.pop(
            "to_torch", None
        )  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataset = cls(**kwargs)
        dataset = (
            torch.utils.data.Subset(dataset, indices)
            if indices is not None
            else dataset
        )
        return dataset.get_loader(
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

    def get_loader(
        self,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
        indices=None,
    ):
        dataset = (
            torch.utils.data.Subset(self, indices) if indices is not None else self
        )
        dataset.to_torch = False  # create dataset with to_torch=False to avoid adding batch dimension twice
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    def timeit(self, num_batches=100, batch_size=1, num_workers=0):
        start = time.time()

        loader = self.get_loader(batch_size=batch_size, num_workers=num_workers)
        for idx, data_blob in enumerate(loader):
            if idx >= num_batches - 1:
                break

        end = time.time()
        print(
            "Total time for loading {} batches: %1.4fs.".format(num_batches)
            % (end - start)
        )
        print("Mean time per batch: %1.4fs." % ((end - start) / num_batches))

    @classmethod
    def write_config(
        cls,
        path,
        dataset_cls_name,
        aug_fcts=None,
        input_size=None,
        to_torch=False,
        updates=None,
        update_strict=False,
        layouts=None,
    ):

        config = {
            "dataset_cls_name": dataset_cls_name,
            "aug_fcts": aug_fcts,
            "input_size": input_size,
            "to_torch": to_torch,
            "updates": updates,
            "update_strict": update_strict,
            "layouts": layouts,
        }

        with open(path, "wb") as file:
            pickle.dump(config, file)

    @classmethod
    def from_config(cls, path, more_updates=None, more_layouts=None, verbose=None):
        with open(path, "rb") as file:
            config = pickle.load(file)

        if more_updates is not None:
            more_updates = (
                [more_updates] if not isinstance(more_updates, list) else more_updates
            )
            updates = config["updates"] if "updates" in config else []
            updates += more_updates
            config["updates"] = updates

        if more_layouts is not None:
            more_layouts = (
                [more_layouts] if not isinstance(more_layouts, list) else more_layouts
            )
            layouts = config["layouts"] if "layouts" in config else []
            layouts += more_layouts
            config["layouts"] = layouts

        if verbose is not None:
            config["verbose"] = verbose

        dataset_cls_name = config["dataset_cls_name"]
        del config["dataset_cls_name"]
        dataset_cls = utils.get_class(dataset_cls_name)
        return dataset_cls(**config)


def _get_paths():
    paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
    with open(paths_file, "r") as paths_file:
        return pytoml.load(paths_file)


def _get_sample_list_path(name):
    sample_lists_path = osp.join(osp.dirname(osp.realpath(__file__)), "sample_lists")
    return osp.join(sample_lists_path, "{}.pickle".format(name))


def _get_path(*keys):
    paths = _get_paths()
    path = None

    for idx, key in enumerate(keys):
        if key in paths:
            if (
                key in paths
                and (isinstance(paths[key], str) or isinstance(paths[key], list))
                and idx == len(keys) - 1
            ):
                path = paths[key]
            else:
                paths = paths[key]

    return path


def _preprocess_sample(sample):

    assert ("depth" in sample or "invdepth" in sample) and (
        not ("depth" in sample and "invdepth" in sample)
    )

    if "depth" in sample:
        with np.errstate(divide="ignore", invalid="ignore"):
            sample["depth"] = sample["depth"].astype(np.float32)
            sample["depth"][sample["depth"] <= 0] = 0
            sample["depth"][~np.isfinite(sample["depth"])] = 0
            sample["invdepth"] = np.nan_to_num(
                1 / sample["depth"], copy=False, nan=0, posinf=0, neginf=0
            )

    elif "invdepth" in sample:
        sample["invdepth"] = sample["invdepth"].astype(np.float32)
        sample["invdepth"][sample["invdepth"] <= 0] = 0
        sample["invdepth"][~np.isfinite(sample["invdepth"])] = 0
        sample["depth"] = np.nan_to_num(
            1 / sample["invdepth"], copy=False, nan=0, posinf=0, neginf=0
        )

    # Calculation of depth range
    if "depth_range" not in sample:
        mask = sample["depth"] > 0
        if mask.any():
            min_depth = np.min(sample["depth"][mask])
            max_depth = np.max(sample["depth"][mask])
            sample["depth_range"] = (min_depth, max_depth)
        else:
            sample["depth_range"] = (0.1, 100.0)

    # This section of the function transforms the poses from keyframe to reference frame.
    # It does this by inverting the transformation from keyframe to reference frame, and then multiplying each pose by this inverted transformation to get the transformation from the reference frame to the keyframe.
    # This is done because the poses are stored in the reference frame, but we want to use them in the keyframe.
    key_idx = sample["keyview_idx"] if "keyview_idx" in sample else 0
    key_to_ref_transform = sample["poses"][key_idx]
    ref_to_key_transform = utils.invert_transform(key_to_ref_transform)
    for idx, to_ref_transform in enumerate(sample["poses"]):
        to_key_transform = np.dot(to_ref_transform, ref_to_key_transform)
        sample["poses"][idx] = to_key_transform
