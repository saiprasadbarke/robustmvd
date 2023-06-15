#!/usr/bin/env python3
import os
import os.path as osp
import argparse
from time import sleep

from tqdm import tqdm


def cp(a, b, verbose=True, followLinks=False):
    os.system('cp -r %s %s "%s" "%s"' % ("-v" if verbose else "", "-L" if followLinks else "", a, b))


def copy_rectified_images(in_base, out_base):
    in_base = osp.join(in_base, "Rectified")
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "images")
        os.makedirs(out_path, exist_ok=True)

        images = sorted([x for x in os.listdir(in_path) if x.endswith("r5000.png")])

        for image in images:
            image_in = osp.join(in_path, image)
            name_split = image.split("_")
            image_corrected = f"{name_split[0]}_{(int(name_split[1]) -1):03d}_{name_split[2]}_{name_split[3]}"
            image_out = osp.join(out_path, image_corrected)
            cp(image_in, image_out)


def copy_gt_depths(in_base, out_base):
    in_base = osp.join(in_base, "dtu", "Depths_raw")
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "gt_depths")
        os.makedirs(out_path, exist_ok=True)

        depths = sorted([x for x in os.listdir(in_path) if x.endswith(".pfm")])

        for idx, depth in enumerate(depths):
            depth_in = osp.join(in_path, depth)
            depth_out = osp.join(out_path, "{:08d}.pfm".format(idx))
            cp(depth_in, depth_out)


def copy_gt_masks(in_base, out_base):
    in_base = osp.join(in_base, "dtu", "Depths_raw")
    scans = os.listdir(in_base)

    for scan in tqdm(scans, "Processed scans"):
        in_path = osp.join(in_base, scan)
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "masks")
        os.makedirs(out_path, exist_ok=True)

        depths = sorted([x for x in os.listdir(in_path) if x.endswith(".png")])

        for idx, depth in enumerate(depths):
            depth_in = osp.join(in_path, depth)
            depth_out = osp.join(out_path, "{:08d}.png".format(idx))
            cp(depth_in, depth_out)


def copy_camera_params(in_base, out_base):
    pair_in = osp.join(in_base, "dtu", "Cameras_1", "pair.txt")
    in_path = osp.join(in_base, "dtu", "Cameras_1")
    cameras = os.listdir(in_path)
    scans = os.listdir(out_base)

    for scan in tqdm(scans, "Processed scans"):
        out_path = osp.join(out_base, scan)
        os.makedirs(out_path, exist_ok=True)
        out_path = osp.join(out_path, "cameras")
        pair_out = osp.join(out_path, "pair.txt")
        print(pair_in, pair_out)
        os.makedirs(out_path, exist_ok=True)
        cp(pair_in, pair_out)
        for camera in cameras:
            cam_in = osp.join(in_path, camera)
            cam_out = osp.join(out_path, camera)
            cp(cam_in, cam_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str)
    parser.add_argument("out_path", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    print("Copying rectified images:")
    copy_rectified_images(args.in_path, args.out_path)

    print("Copying GT depths:")
    copy_gt_depths(args.in_path, args.out_path)

    print("Copying GT masks:")
    copy_gt_masks(args.in_path, args.out_path)

    print("Copying camera parameters:")
    copy_camera_params(args.in_path, args.out_path)

    print("Done")
