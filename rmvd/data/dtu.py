import os
import os.path as osp
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

from .dataset import Dataset, Sample, _get_sample_list_path
from .registry import register_dataset, register_default_dataset
from .layouts import MVDUnstructuredDefaultLayout, AllImagesLayout
from rmvd.exceptions.io_exceptions import PFMFileException

# Taken from https://github.com/xy-guo/MVSNet_pytorch/blob/master/lists/dtu/train.txt
DTU_TRAIN_SCENES = [
    "scan2",
    "scan6",
    "scan7",
    "scan8",
    "scan14",
    "scan16",
    "scan18",
    "scan19",
    "scan20",
    "scan22",
    "scan30",
    "scan31",
    "scan36",
    "scan39",
    "scan41",
    "scan42",
    "scan44",
    "scan45",
    "scan46",
    "scan47",
    "scan50",
    "scan51",
    "scan52",
    "scan53",
    "scan55",
    "scan57",
    "scan58",
    "scan60",
    "scan61",
    "scan63",
    "scan64",
    "scan65",
    "scan68",
    "scan69",
    "scan70",
    "scan71",
    "scan72",
    "scan74",
    "scan76",
    "scan83",
    "scan84",
    "scan85",
    "scan87",
    "scan88",
    "scan89",
    "scan90",
    "scan91",
    "scan92",
    "scan93",
    "scan94",
    "scan95",
    "scan96",
    "scan97",
    "scan98",
    "scan99",
    "scan100",
    "scan101",
    "scan102",
    "scan103",
    "scan104",
    "scan105",
    "scan107",
    "scan108",
    "scan109",
    "scan111",
    "scan112",
    "scan113",
    "scan115",
    "scan116",
    "scan119",
    "scan120",
    "scan121",
    "scan122",
    "scan123",
    "scan124",
    "scan125",
    "scan126",
    "scan127",
    "scan128",
]

# Taken from https://github.com/xy-guo/MVSNet_pytorch/blob/master/lists/dtu/val.txt
DTU_VAL_SCENES = [
    "scan3",
    "scan5",
    "scan17",
    "scan21",
    "scan28",
    "scan35",
    "scan37",
    "scan38",
    "scan40",
    "scan43",
    "scan56",
    "scan59",
    "scan66",
    "scan67",
    "scan82",
    "scan86",
    "scan106",
    "scan117",
]


DTU_EVAL_SCENES = [
    "scan1",
    "scan4",
    "scan9",
    "scan10",
    "scan11",
    "scan12",
    "scan13",
    "scan15",
    "scan23",
    "scan24",
    "scan29",
    "scan32",
    "scan33",
    "scan34",
    "scan48",
    "scan49",
    "scan62",
    "scan75",
    "scan77",
    "scan110",
    "scan114",
    "scan118",
]


def readPFM(file):
    file = open(file, "rb")

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise PFMFileException("Not a PFM file.")

    if dim_match := re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii")):
        width, height = list(map(int, dim_match.groups()))
    else:
        raise PFMFileException("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    endian = "<" if scale < 0 else ">"

    data = np.fromfile(file, f"{endian}f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    file.close()
    return data


def load_image(root, path):
    """This function loads an image from disk and returns it as a numpy array"""
    path = f"images/rect_{path[0]:03d}_{path[1]}_r5000.png"
    img_path = osp.join(root, path)
    img = np.array(Image.open(img_path))  # H,W,3 ; dtype np.uint8
    img = img.transpose(2, 0, 1).astype(np.float32)  # 3,H,W ; dtype np.uint8
    return img


def load_pose(root, path):
    """This function loads the camera pose from a file on the disk and returns it as a numpy array"""
    path = f"cameras/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        pose_lines = [x[:-1] for x in pose_file.readlines()][1:5]
        pose_elements = [float(x) for line in pose_lines for x in line.split()]
        pose_matrix = np.array(
            [
                pose_elements[:4],
                pose_elements[4:8],
                pose_elements[8:12],
                pose_elements[12:16],
            ],
            dtype=np.float32,
        )
    return pose_matrix  # 4, 4


def load_intrinsics(root, path):
    """This function loads camera intrincs from a file on the disk and returns it as a numpy array"""
    path = f"cameras/{path:08d}_cam.txt"
    pose_path = osp.join(root, path)
    with open(pose_path) as pose_file:
        intrinsic_lines = [x[:-1] for x in pose_file.readlines()][7:10]
        intrinsic_elements = [float(x) for line in intrinsic_lines for x in line.split()]
        intrinsic_matrix = np.array(
            [
                intrinsic_elements[:3],
                intrinsic_elements[3:6],
                intrinsic_elements[6:9],
            ],
            dtype=np.float32,
        )
    return intrinsic_matrix  # 3, 3


def load_depth(root, path):
    """This function loads rendered depth maps from a file on the disk and returns it as a numpy array"""
    path = f"gt_depths/{path:08d}.pfm"
    depth = readPFM(osp.join(root, path))
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)  # Replace NaNs with 0.0
    depth = np.expand_dims(depth, 0).astype(np.float32)  # (1,H,W)
    return depth  # 1, H, W, np.float32


def load_mask(root, path):
    """This function loads the mask images and returns it as a numpy array"""
    path = f"masks/{path:08d}.png"
    mask_path = osp.join(root, path)
    mask = np.array(Image.open(mask_path))  # H,W ; dtype np.uint8
    mask = np.expand_dims(mask, 0).astype(np.float32)  # 1, H, W
    return mask


def load(key, root, val):
    """This function is a general function that dispatches to the appropriate load function above based on the key argument. If the value of the key is a list, then each element of the list is recursively loaded using the same load function. Otherwise, the appropriate load function is called based on the key, and the result is returned."""
    if isinstance(val, list):
        return [load(key, root, v) for v in val]
    else:
        if key == "images":
            return load_image(root, val)
        elif key == "depth":
            return load_depth(root, val)
        elif key == "intrinsics":
            return load_intrinsics(root, val)
        elif key == "poses":
            return load_pose(root, val)
        elif key == "masks":
            return load_mask(root, val)
        else:
            return val


class DTUPair:
    def __init__(self, path):
        with open(path) as pair_file:
            pair_lines = pair_file.readlines()
            self.keyview_ids = [int(x.rstrip()) for x in pair_lines[1::2]]
            pair_lines = [x.rstrip() for x in pair_lines[2::2]]
            pair_lines = [x.split(" ") for x in pair_lines]
            pair_indices = [pair_line[1::2] for pair_line in pair_lines]
            self._other_view_ids = [list(map(int, indices)) for indices in pair_indices]
            pair_scores = [pair_line[2::2] for pair_line in pair_lines]
            self._other_view_scores = [list(map(float, scores)) for scores in pair_scores]

            for idx, other_view_ids in enumerate(self._other_view_ids):
                while 0 < len(other_view_ids) < 10:
                    other_view_scores = self._other_view_scores[idx]

                    to_be_added = min(len(other_view_ids), 10 - len(other_view_ids))
                    other_view_ids += other_view_ids[:to_be_added]
                    other_view_scores += other_view_scores[:to_be_added]

                    self._other_view_ids[idx] = other_view_ids
                    self._other_view_scores[idx] = other_view_scores

    def get_source_ids(self, keyview_id):
        idx = self.keyview_ids.index(keyview_id)
        return self._other_view_ids[idx]

    def get_source_scores(self, keyview_id):
        idx = self.keyview_ids.index(keyview_id)
        return self._other_view_scores[idx]


class DTUSample(Sample):
    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.data = {}

    def load(self, root):
        base = osp.join(root, self.base)
        out_dict = {"_base": base, "_name": self.name}

        for key, val in self.data.items():
            out_dict[key] = load(key, base, val)

        return out_dict


class DTUMinDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        pose_path = osp.join(root, self.path)
        with open(pose_path) as pose_file:
            depth_line = pose_file.readlines()[11]
            depths = [float(x) for x in depth_line.split(" ")]
            min_depth, max_depth = depths[0], depths[-1]
        return min_depth  # float value


class DTUMaxDepth:
    def __init__(self, path):
        self.path = path

    def load(self, root):
        pose_path = osp.join(root, self.path)
        with open(pose_path) as pose_file:
            depth_line = pose_file.readlines()[11]
            depths = [float(x) for x in depth_line.split(" ")]
            min_depth, max_depth = depths[0], depths[-1]
        return max_depth  # float value


class DTUImage:
    def __init__(self, path: str):
        self.path = path

    def load(self, root: str) -> np.ndarray:
        path = osp.join(root, self.path)
        return np.array(Image.open(path), dtype=np.float32).transpose(2, 0, 1)


class DTUDepth:
    def __init__(self, path, format=None):
        self.path = path

    def load(self, root):
        depth = readPFM(osp.join(root, self.path)) / 1000
        depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)
        depth = np.expand_dims(depth, 0)  # 1HW
        return depth


class DTUScene:
    def __init__(self, root):
        self.root = root
        self.name = osp.split(root)[1]

        pair = DTUPair(osp.join(root, "cameras", "pair.txt"))
        self.source_ids = {keyview_id: pair.get_source_ids(keyview_id) for keyview_id in pair.keyview_ids}
        self.source_scores = {keyview_id: pair.get_source_scores(keyview_id) for keyview_id in pair.keyview_ids}

        cam_files = [x for x in os.listdir(osp.join(root, "cameras")) if x.endswith("cam.txt")]
        self.min_depths = {int(x[:8]): DTUMinDepth(osp.join("cameras", x)).load(root) for x in cam_files}
        self.max_depths = {int(x[:8]): DTUMaxDepth(osp.join("cameras", x)).load(root) for x in cam_files}

        images = [x for x in os.listdir(osp.join(root, "images")) if x.endswith("0_r5000.png")]
        self.images = [int(x.split("_")[1]) for x in images]
        masks = [x for x in os.listdir(osp.join(root, "masks")) if x.endswith(".png")]
        self.masks = [int(x[:8]) for x in masks]
        self.masks = sorted(self.masks)[: len(self.images)]
        depths = [x for x in os.listdir(osp.join(root, "gt_depths")) if x.endswith(".pfm")]
        self.depths = [int(x[:8]) for x in depths]
        self.depths = sorted(self.depths)[: len(self.images)]
        self.intrinsics = [int(x[:8]) for x in cam_files]
        self.poses = [int(x[:8]) for x in cam_files]

        assert len(
            set(self.images)
            .intersection(set(self.depths))
            .intersection(set(self.intrinsics))
            .intersection(set(self.min_depths.keys()))
            .intersection(set(self.max_depths.keys()))
            .intersection(set(self.poses))
        ) == len(
            self.images
        ), f"{len(set(self.depths)).intersection(set(self.intrinsics)).intersection(set(self.min_depths.keys())).intersection(set(self.max_depths.keys())).intersection(set(self.poses))} not equal to {len(self.images)}"

        for key_id, cur_source_ids in self.source_ids.items():
            assert key_id in self.images
            assert key_id in self.depths
            assert key_id in self.poses
            assert key_id in self.intrinsics
            for source_id in cur_source_ids:
                assert source_id in self.images
                assert source_id in self.depths
                assert source_id in self.poses
                assert source_id in self.intrinsics
            assert len(cur_source_ids) == 10

    def __len__(self):
        return len(self.images)


class DTU(Dataset):
    base_dataset = "dtu"

    def _init_samples(self, scene_names=None, num_source_views=None, all_combinations=True):
        sample_list_path = _get_sample_list_path(self.name)
        if sample_list_path is not None and osp.isfile(sample_list_path):
            super()._init_samples_from_list()
        else:
            self._init_samples_from_root_dir(
                scene_names=scene_names, num_source_views=num_source_views, all_combinations=all_combinations
            )
            self._write_samples_list()

    def _init_samples_from_root_dir(self, scene_names=None, num_source_views=None, all_combinations=True):
        from itertools import combinations

        scenes = [x for x in os.listdir(self.root) if osp.isdir(osp.join(self.root, x))]
        scenes = [x for x in scenes if x in scene_names] if scene_names is not None else scenes
        scenes = sorted(scenes)
        scenes = [DTUScene(osp.join(self.root, x)) for x in scenes]

        for scene in tqdm(scenes) if self.verbose else scenes:
            for key_id in scene.source_ids.keys():
                all_source_ids = scene.source_ids[key_id]
                all_scores = scene.source_scores[key_id]
                cur_num_source_views = num_source_views if num_source_views is not None else len(all_source_ids)
                if all_combinations:
                    source_id_combinations = [list(x) for x in list(combinations(all_source_ids, cur_num_source_views))]
                else:
                    source_id_combinations = [all_source_ids[:cur_num_source_views]]
                for light_idx in range(7):
                    for source_ids in source_id_combinations:
                        sample = DTUSample(
                            name=scene.name + "/key{:02d}".format(key_id) + f"/light{light_idx}",
                            base=scene.name,
                        )
                        all_ids = [key_id] + source_ids
                        images = [(x, light_idx) for x in all_ids]
                        poses = all_ids
                        masks = key_id
                        intrinsics = all_ids
                        min_depth = scene.min_depths[key_id]
                        max_depth = scene.max_depths[key_id]
                        depth = key_id

                        sample.data["images"] = images
                        sample.data["poses"] = poses
                        sample.data["intrinsics"] = intrinsics
                        sample.data["masks"] = masks
                        sample.data["depth"] = depth
                        sample.data["depth_range"] = (min_depth, max_depth)
                        sample.data["keyview_idx"] = 0

                        # sample.data['_keyview_id'] = key_id
                        # sample.data['_source_view_ids'] = source_ids
                        # sample.data['_source_view_scores'] = source_scores

                        self.samples.append(sample)


@register_default_dataset
class DTURobustMVD(DTU):
    split = "robustmvd"
    dataset_type = "mvd"

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("dtu", "root")
        scene_names = DTU_TRAIN_SCENES
        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=11, max_views=4),
            AllImagesLayout("all_images", num_views=11),
        ]
        layouts = default_layouts + layouts if layouts is not None else default_layouts

        super().__init__(
            scene_names=scene_names,
            num_source_views=2,
            root=root,
            layouts=layouts,
            **kwargs,
        )


@register_dataset
class DTUMvsnetTrain(DTU):
    split = "train_mvsnet"
    dataset_type = "mvd"

    def __init__(self, root=None, layouts=None, **kwargs):
        root = root if root is not None else self._get_path("dtu", "root")
        scene_names = DTU_TRAIN_SCENES
        default_layouts = [
            MVDUnstructuredDefaultLayout("default", num_views=11, max_views=4),
            AllImagesLayout("all_images", num_views=11),
        ]
        layouts = default_layouts + (layouts if layouts is not None else default_layouts)

        super().__init__(
            scene_names=scene_names, num_source_views=2, all_combinations=False, root=root, layouts=layouts, **kwargs
        )
