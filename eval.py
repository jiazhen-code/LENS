import os
import sys

# Make the third-party LISA tree importable (it provides the top-level `model_lisa`
# package used by lens.data) and keep the repo root on sys.path (for `lens`).
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "LISA"))

from lens.data.test_dataset import TestReasoningDataset, TestReferDataset, collate_fn_test
from lens.data.trainval_dataset import collate_fn_val
import torch
from functools import partial
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.distributed as dist
import random
from tqdm import tqdm
from lens.utils.inference import data_deal
from lens.utils.visualize import save_seg_visualization
from transformers import CLIPImageProcessor
from PIL import Image
import torch.distributed as dist

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def prepare_input(input_dict, precision, is_cuda=True):
    """Prepare input data based on precision."""
    if precision == "fp16":
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
    elif precision == "bf16":
        input_dict["images"] = input_dict["images"].bfloat16()
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
    else:
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_clip"] = input_dict["images_clip"].float()
    if is_cuda:
        input_dict = dict_to_cuda(input_dict)
    return input_dict

def random_seed(seed = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _attn_pad_frac(model, image):
    """Content fraction (fx, fy) of the padded square a 'pad'-mode backbone (Qwen/LLaVA-OneVision)
    squares the image into, so the viz can crop the padded attention heatmap and rescale
    keypoints back to the original-image frame. None for stretch / LLaVA (image fills frame)."""
    m = getattr(model, "module", model)  # unwrap DDP / accelerate
    bb = getattr(getattr(m, "conditioner", None), "backbone", None)
    if bb is None or getattr(bb, "image_hw", None) is None:
        return None
    if getattr(bb, "image_square_mode", "pad") != "pad":
        return None
    h, w = np.asarray(image).shape[:2]
    mx = max(h, w)
    return (w / mx, h / mx)


from torchvision.transforms import ToPILImage
from tqdm import tqdm
@torch.inference_mode()
def validate(accelerator, val_loader, full, global_iters, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # ReasonSeg short/long split (long = sentence/reasoning query, short = phrase). Routed per
    # sample by its is_sentence flag and accumulated as fixed-shape numpy sums so the cross-rank
    # reduction is safe even if a rank sees zero samples of one subset.
    sub_keys = ("short", "long")
    sub_inter = {k: np.zeros(2) for k in sub_keys}
    sub_union = {k: np.zeros(2) for k in sub_keys}
    sub_giou_sum = {k: np.zeros(2) for k in sub_keys}
    sub_count = {k: 0 for k in sub_keys}

    full.eval()

    aaa = 0
    conversation_records = {}
    for input_dict in tqdm(val_loader):
    # for input_dict in val_loader:
        torch.cuda.empty_cache()
        batch = prepare_input(input_dict, args.precision, is_cuda=True)

        imgs = batch['images']
        offset = batch['offset']
        images_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_i = (
                imgs[i]
                .unsqueeze(0)
                .expand(end_i - start_i, -1, -1, -1)
                .contiguous()
            )
            images_list.append(images_i)
        imgs = torch.cat(images_list, dim=0)

        mask_raw = batch['masks_list']
        mask_raw_ = []
        ori_size = []
        resized_size = []
        raw_img = []
        mask_trans = []
        imgs_raw_path = []
        for s, m in enumerate(mask_raw):
            resize_list = batch['sam_mask_shape_list'][s][0]
            for mm in m:
                mask_raw_.append((mm.int()).to(imgs.device))
                mask_tensor, _, _ = data_deal(mm.cpu().numpy(), imgs[i].shape[-1], is_mask=True)
                ori_size.append([mm.shape[0], mm.shape[1]])
                rl = [resize_list[1], resize_list[0]]
                resized_size.append(rl)
                img_r = Image.open(batch['image_paths'][s]).convert('RGB')
                img_r, _, _ = data_deal(img_r, imgs[i].shape[-1], is_mask=True)
                raw_img.append(img_r)
                imgs_raw_path.append(batch['image_paths'][s])
                mask_trans.append(((mask_tensor // 255).int()).to(imgs.device))
        mask_raw = mask_raw_
        prompts = batch['conversation_list']

        imgs_pil = [ToPILImage()(i) for i in raw_img]
        # imgs_raw_path = [i for i in raw_img]
        head_maps, keypoint_coords, pred_masks, loss_seg = full(imgs, imgs_pil, prompts, mask_raw, resized_size, ori_size)

        masks_list = [m.int().unsqueeze(0) for m in mask_raw]
        output_list = [(p.sigmoid() > 0.6).int() for p in pred_masks]
        # masks_list = mask_raw[0].int().unsqueeze(0)
        # output_list = (pred_masks[0].sigmoid() > 0.6).int()
        conversation_records.update(input_dict['conversation_records'][0])
        # assert len(pred_masks) == 1
        ddd = str(dist.get_rank())
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            if ddd == "7" and aaa == 22:
                aaaaaaa = 1
            mask_i = mask_i.int()
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
            
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=len(masks_list))

        # Route this sample's metrics to its ReasonSeg subset. val_batch_size==1, so one
        # is_sentence flag governs all masks of the batch; same accumulation as the overall
        # meters (cIoU = sum(inter)/sum(union); gIoU = sum(per-mask acc_iou)/sum(masks)).
        sub = "long" if bool(batch.get("is_sentence_list", [False])[0]) else "short"
        sub_inter[sub] += intersection
        sub_union[sub] += union
        sub_giou_sum[sub] += acc_iou * len(masks_list)
        sub_count[sub] += len(masks_list)

        # Save a local visualization (image + attention overlay + predicted keypoints,
        # pred mask vs. GT mask) plus sidecar .txt (instruction) and .npy (raw attention).
        # During training this runs every epoch via eval_model(); capped at args.max_viz.
        if getattr(args, "save_viz", False) and aaa < getattr(args, "max_viz", 20):
            # Re-open the first sample's image at original resolution (matches the masks)
            # so the attention heatmap and keypoints can be overlaid on it.
            viz_image = np.array(Image.open(imgs_raw_path[0]).convert("RGB"))
            save_seg_visualization(
                pred_mask=output_list[0].cpu().numpy()[0],
                gt_mask=(masks_list[0][0].cpu().numpy() == 1),
                keypoint_coords=keypoint_coords[0:1].cpu().detach(),
                acc_iou=acc_iou,
                save_path=f"viz/step_{global_iters}/rank_{ddd}/{aaa}.png",
                attention_map=head_maps[0],
                prompt=prompts[0],
                image=viz_image,
                meta={"image_path": imgs_raw_path[0]},
                attn_pad_frac=_attn_pad_frac(full, viz_image),
            )
        aaa += 1

    # all reduce in distributed setting
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    if ddd == '0':
        print(intersection_meter.sum[1], union_meter.sum[1])
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    total_samples = int(acc_iou_meter.count)

    # ReasonSeg short/long subset reduction. All ranks must enter the all_reduce together, so
    # gate on the dataset name (identical across ranks), never on rank.
    subset_results = None
    if args.val_dataset == "ReasonSeg":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _reduce(arr):
            t = torch.as_tensor(arr, dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.cpu().numpy()

        subset_results = {}
        for k in sub_keys:
            inter, union = _reduce(sub_inter[k]), _reduce(sub_union[k])
            giou_sum = _reduce(sub_giou_sum[k])
            cnt = float(_reduce(np.array([float(sub_count[k])]))[0])
            subset_results[k] = (
                float(giou_sum[1] / (cnt + 1e-5)),     # gIoU
                float(inter[1] / (union[1] + 1e-10)),  # cIoU
                int(cnt),                              # N
            )

    if accelerator.is_main_process:
        # print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        if args.use_wandb:
            wandb.log({"giou": giou}, step=global_iters)
            wandb.log({"ciou": ciou}, step=global_iters)
            wandb.log({"nums": total_samples}, step=global_iters)
        else:
            writer.add_scalar("giou", giou, global_iters)
            writer.add_scalar("ciou", ciou, global_iters)
            writer.add_scalar("nums", total_samples, global_iters)

        if subset_results is not None:
            sh, lo = subset_results["short"], subset_results["long"]
            print(f"ReasonSeg [{args.val_split}]  (gIoU / cIoU / N)")
            print(f"  short  : {sh[0]:.4f} / {sh[1]:.4f} / {sh[2]}")
            print(f"  long   : {lo[0]:.4f} / {lo[1]:.4f} / {lo[2]}")
            print(f"  overall: {giou:.4f} / {ciou:.4f} / {total_samples}")
            if args.use_wandb:
                wandb.log({"giou_short": sh[0], "ciou_short": sh[1],
                           "giou_long": lo[0], "ciou_long": lo[1]}, step=global_iters)
            else:
                writer.add_scalar("giou_short", sh[0], global_iters)
                writer.add_scalar("ciou_short", sh[1], global_iters)
                writer.add_scalar("giou_long", lo[0], global_iters)
                writer.add_scalar("ciou_long", lo[1], global_iters)

    # conditioner.train()
    # net.train()
    return giou, ciou, total_samples, conversation_records


from lens.config import EvalConfig


class MyArgs:
    """Runtime view of :class:`lens.config.EvalConfig`.

    Holds the same attributes as before (so ``validate`` / ``eval_model`` are unchanged),
    but every value now comes from an ``EvalConfig`` instead of being hard-coded. Calling
    ``MyArgs()`` with no argument yields the original defaults.
    """

    def __init__(self, cfg: EvalConfig = None):
        if cfg is None:
            cfg = EvalConfig()
        self.val_dataset = cfg.val_dataset
        self.val_split = cfg.val_split
        self.dataset_dir = cfg.dataset_dir
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.clip_vision_model)
        self.image_size = cfg.image_size
        self.eval_only = cfg.eval_only
        self.val_batch_size = cfg.val_batch_size
        self.use_wandb = cfg.use_wandb
        self.precision = cfg.precision
        self.use_mm_start_end = cfg.use_mm_start_end
        self.exp_name = cfg.exp_name
        self.log_dir = cfg.log_dir
        self.workers = cfg.workers
        # Local visualization saved under viz/step_<step>/rank_<rank>/<idx>.png.
        self.save_viz = cfg.save_viz
        self.max_viz = cfg.max_viz


def eval_model(accelerator, full, steps, args=None, eval_cfg: EvalConfig = None):
    if args is None:
        args = MyArgs(eval_cfg)
    conversation_records = {}
    reason_seg_dataset = ["ReasonSeg"]
    refer_seg_dataset = [
        "refcoco", "refcoco+", "refcocog",
    ]
    if args.val_dataset in reason_seg_dataset:
        val_dataset = TestReasoningDataset(
            args.dataset_dir,
            args.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split=args.val_split,  # val|test
            eval_only=args.eval_only,
            conversation_records=conversation_records
        )
    elif args.val_dataset in refer_seg_dataset:
        val_dataset = TestReferDataset(
            args.dataset_dir,
            args.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split=args.val_split,  # val|test|testA|testB
            conversation_records=conversation_records
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.val_dataset}")

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #     val_dataset, shuffle=False, drop_last=False
        # )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            # sampler=val_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=None,
                use_mm_start_end=args.use_mm_start_end
            ),
        )
        val_loader = accelerator.prepare(val_loader)
    if accelerator.is_main_process:
        if args.use_wandb:
            import wandb
            wandb.init(project="read", name=args.exp_name)
        else:
            writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    giou, ciou, total_samples, conversation_records = validate(accelerator, val_loader, full, steps, writer, args)
    torch.cuda.empty_cache()
    return giou, ciou, total_samples, conversation_records


if __name__ == '__main__':
    import argparse
    from accelerate import Accelerator
    from lens.config import load_config
    from lens.models import load_full_state_dict
    from lens.models.decoder import Decoder, sigmoid_ce_loss, dice_loss
    from lens.models.conditioner import MLLM_Conditioner, FullModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/qwen2vl_2b.yaml",
                        help="Path to the experiment YAML config.")
    parser.add_argument("--checkpoint", type=str, default="/ssd1/save_dir_qwen/full_10.pth",
                        help="FullModel checkpoint to evaluate.")
    cli_args = parser.parse_args()
    cfg = load_config(cli_args.config)

    accelerator = Accelerator()
    # if accelerator.is_main_process:
    #     SAM_CHECKPOINT = download_sam_checkpoint()
    accelerator.wait_for_everyone()

    # clip_encoder = Conditioner().to(accelerator.device).eval()
    conditioner = MLLM_Conditioner(cfg.model).to(accelerator.device)
    net = Decoder(cfg.model.decoder)

    full = FullModel(net, conditioner, kp_thresh=cfg.model.kp_thresh).to(torch.bfloat16)
    load_full_state_dict(full, cli_args.checkpoint, map_location='cpu')
    full = accelerator.prepare(
        full
    )



    giou, ciou, nums, cr = eval_model(accelerator, full, -2, eval_cfg=cfg.eval)
    if accelerator.is_main_process:
        print(f"Evaluation Results:")
        print(f"nums: {nums}, gIoU: {giou:.4f}, cIoU: {ciou:.4f}")

    accelerator.wait_for_everyone()