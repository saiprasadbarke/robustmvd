# Standard Imports

# Local Imports
from .registry import register_loss
from .utils import bin_op_reduce

# External Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


@register_loss
class VismvnsetMultiscaleMultiviewAggregate(nn.Module):  # TODO
    def __init__(
        self,
        model,
    ):
        super().__init__()

    @property
    def name(self):
        return type(self).__name__

    def forward(self, sample_inputs, sample_gt, pred, aux, iteration):
        # def forward(
        #     self, outputs, gt, masks, ref_cam, max_d, occ_guide=False, mode="soft"
        # ):  # MVS
        outputs = aux["outputs"]
        refined_depth = pred["depth"]
        gt = sample_gt["depth"]
        masks = sample_inputs["masks"]
        max_d = 192
        mode = "soft"
        occ_guide = False
        ref_cam = aux["ref_cam"]
        depth_start = ref_cam[:, 1:2, 3:4, 0:1]  # n111
        depth_interval = ref_cam[:, 1:2, 3:4, 1:2]  # n111
        depth_end = depth_start + (max_d - 2) * depth_interval  # strict range

        stage_losses = []
        stats = []
        for est_depth, pair_results in outputs:
            gt_downsized = F.interpolate(
                gt,
                size=(est_depth.size()[2], est_depth.size()[3]),
                mode="bilinear",
                align_corners=False,
            )
            masks_downsized = [
                F.interpolate(
                    mask,
                    size=(est_depth.size()[2], est_depth.size()[3]),
                    mode="nearest",
                )
                for mask in masks
            ]
            in_range = torch.min(
                (gt_downsized >= depth_start), (gt_downsized <= depth_end)
            )
            masks_valid = [
                torch.min((mask > 50), in_range) for mask in masks_downsized
            ]  # mask and in_range
            masks_overlap = [
                torch.min((mask > 200), in_range) for mask in masks_downsized
            ]
            union_overlap = bin_op_reduce(masks_overlap, torch.max)  # A(B+C)=AB+AC
            valid = union_overlap if occ_guide else in_range

            same_size = (
                est_depth.size()[2] == pair_results[0][0].size()[2]
                and est_depth.size()[3] == pair_results[0][0].size()[3]
            )
            gt_interm = (
                gt_downsized
                if same_size
                else F.interpolate(
                    gt,
                    size=(pair_results[0][0].size()[2], pair_results[0][0].size()[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            masks_interm = (
                masks_downsized
                if same_size
                else [
                    F.interpolate(
                        mask,
                        size=(
                            pair_results[0][0].size()[2],
                            pair_results[0][0].size()[3],
                        ),
                        mode="nearest",
                    )
                    for mask in masks
                ]
            )
            in_range_interm = (
                in_range
                if same_size
                else torch.min((gt_interm >= depth_start), (gt_interm <= depth_end))
            )
            masks_valid_interm = (
                masks_valid
                if same_size
                else [torch.min((mask > 50), in_range_interm) for mask in masks_interm]
            )  # mask and in_range
            masks_overlap_interm = (
                masks_overlap
                if same_size
                else [torch.min((mask > 200), in_range_interm) for mask in masks_interm]
            )
            union_overlap_interm = (
                union_overlap
                if same_size
                else bin_op_reduce(masks_overlap_interm, torch.max)
            )  # A(B+C)=AB+AC
            valid_interm = (
                valid
                if same_size
                else union_overlap_interm
                if occ_guide
                else in_range_interm
            )

            abs_err = (est_depth - gt_downsized).abs()
            abs_err_scaled = abs_err / depth_interval
            pair_abs_err = [
                (est - gt_interm).abs()
                for est in [est for est, (uncert, occ) in pair_results]
            ]
            pair_abs_err_scaled = [err / depth_interval for err in pair_abs_err]

            l1 = abs_err_scaled[valid].mean()

            # ===== pair l1 =====
            if occ_guide:
                pair_l1_losses = [
                    err[mask_overlap].mean()
                    for err, mask_overlap in zip(
                        pair_abs_err_scaled, masks_overlap_interm
                    )
                ]
            else:
                pair_l1_losses = [
                    err[in_range_interm].mean() for err in pair_abs_err_scaled
                ]
            pair_l1_loss = sum(pair_l1_losses) / len(pair_l1_losses)

            # ===== uncert =====
            if mode in ["soft", "hard", "uwta"]:
                if occ_guide:
                    uncert_losses = [
                        (
                            err[mask_valid] * (-uncert[mask_valid]).exp()
                            + uncert[mask_valid]
                        ).mean()
                        for err, (est, (uncert, occ)), mask_valid, mask_overlap in zip(
                            pair_abs_err_scaled,
                            pair_results,
                            masks_valid_interm,
                            masks_overlap_interm,
                        )
                    ]
                else:
                    uncert_losses = [
                        (
                            err[in_range_interm] * (-uncert[in_range_interm]).exp()
                            + uncert[in_range_interm]
                        ).mean()
                        for err, (est, (uncert, occ)) in zip(
                            pair_abs_err_scaled, pair_results
                        )
                    ]
                uncert_loss = sum(uncert_losses) / len(uncert_losses)

            # ===== logistic =====
            if occ_guide and mode in ["soft", "hard", "uwta"]:
                logistic_losses = [
                    nn.SoftMarginLoss()(
                        occ[mask_valid], -mask_overlap[mask_valid].to(gt.dtype) * 2 + 1
                    )
                    for (est, (uncert, occ)), mask_valid, mask_overlap in zip(
                        pair_results, masks_valid_interm, masks_overlap_interm
                    )
                ]
                logistic_loss = sum(logistic_losses) / len(logistic_losses)

            less1 = (abs_err_scaled[valid] < 1.0).to(gt.dtype).mean()
            less3 = (abs_err_scaled[valid] < 3.0).to(gt.dtype).mean()

            pair_loss = pair_l1_loss
            if mode in ["soft", "hard", "uwta"]:
                pair_loss = pair_loss + uncert_loss
                if occ_guide:
                    pair_loss = pair_loss + logistic_loss
            loss = l1 + pair_loss
            stage_losses.append(loss)
            stats.append((l1, less1, less3))

        abs_err = (refined_depth - gt_downsized).abs()
        abs_err_scaled = abs_err / depth_interval
        l1 = abs_err_scaled[valid].mean()
        less1 = (abs_err_scaled[valid] < 1.0).to(gt.dtype).mean()
        less3 = (abs_err_scaled[valid] < 3.0).to(gt.dtype).mean()

        loss = (
            stage_losses[0] * 0.5 + stage_losses[1] * 1.0 + stage_losses[2] * 2.0
        )  # + l1*2.0

        # return loss, pair_loss, less1, less3, l1, stats, abs_err_scaled, valid
        return loss, {}, {}
