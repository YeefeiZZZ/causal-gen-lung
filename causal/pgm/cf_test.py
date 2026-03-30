import argparse
import copy
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from dscm import DSCM
from flow_pgm import FlowPGM, ChestPGM
from layers import TraceStorage_ELBO
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_pgm import eval_epoch, preprocess, sup_epoch
from utils_pgm import plot_cf, update_stats, plot_cf_mimic

from train_pgm import setup_dataloaders, setup_dataloaders_cf
sys.path.append("..")
from datasets import get_attr_max_min
from hps import Hparams
from train_setup import setup_directories, setup_logging, setup_tensorboard
from utils import EMA, seed_all
from vae import HVAE

import gc

def loginfo(title: str, logger: Any, stats: Dict[str, Any]):
    logger.info(f"{title} | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))


def inv_preprocess(pa: Dict[str, Tensor]) -> Dict[str, Tensor]:
    # undo [-1,1] parent preprocessing back to original range
    for k, v in pa.items():
        # if k != "mri_seq" and k != "sex":
        if k == "age":
            pa[k] = (v + 1) / 2  # [-1,1] -> [0,1]
            # _max, _min = get_attr_max_min(k)
            _max, _min = 73, 44
            pa[k] = pa[k] * (_max - _min) + _min
    return pa

def save_plot(
    save_path: str,
    obs: Dict[str, Tensor],
    cfs: Dict[str, Tensor],
    do: Dict[str, Tensor],
    var_cf_x: Optional[Tensor] = None,
    num_images: int = 10,
    step: int = 0,
    is_train: bool = False,
    epoch: int = 0
) -> None:
    _ = plot_cf_mimic(
        obs["x"],
        cfs["x"],
        inv_preprocess({k: v for k, v in obs.items() if k != "x"}),  # pa
        inv_preprocess({k: v for k, v in cfs.items() if k != "x"}),  # cf_pa
        inv_preprocess(do),
        var_cf_x,  # counterfactual variance per pixel
        num_images=num_images,
        step = step,
        is_train = is_train
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_metrics(
    dataset: str, preds: Dict[str, List[Tensor]], targets: Dict[str, List[Tensor]]
) -> Dict[str, Tensor]:
    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k]).squeeze().cpu()
    stats = {}
    for k in preds.keys():
        if "ukbb" in dataset:
            if k == "mri_seq" or k == "sex":
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            else:  # continuous variables
                preds_k = (preds[k] + 1) / 2  # [-1,1] -> [0,1]
                _max, _min = get_attr_max_min(k)
                preds_k = preds_k * (_max - _min) + _min
                norm = 1000 if "volume" in k else 1  # for volume in ml
                stats[k + "_mae"] = (targets[k] - preds_k).abs().mean().item() / norm
        elif "mimic" in dataset:
            if k in ["sex"]:
                
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(), preds[k].numpy(), average="macro"
                )
                stats[k + "_acc"] = (
                    targets[k] == torch.round(preds[k])
                ).sum().item() / targets[k].shape[0]
            elif k == "age":
                preds_k = (preds[k] + 1) * 50  # unormalize
                targets_k = (targets[k] + 1) * 50  # unormalize
                
                # print("-----------------------------------------------------")
                # print(preds_k, ",", targets_k)
                # print("-----------------------------------------------------")
                
                
                stats[k + "_mae"] = (targets_k - preds_k).abs().mean().item()
            elif k in ["finding", "race"]:
                num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
                stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
                stats[k + "_rocauc"] = roc_auc_score(
                    targets[k].numpy(),
                    preds[k].numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        else:
            NotImplementedError
    return stats


def cf_epoch(
    args: Hparams,
    model: nn.Module,
    ema: nn.Module,
    dataloaders: Dict[str, DataLoader],
    elbo_fn: TraceStorage_ELBO,
    optimizers: Optional[Tuple] = None,
    split: str = "train",
    epoch : int = 0
):
    "counterfactual auxiliary training/eval epoch"
    is_train = split == "train"
    if is_train:
        print("ZZZZZZZZZZZZZZZZZZZZSWITCH TO TRAIN MODEZZZZZZZZZZZZZZZZZZZZZ")
    else:
        print("ZZZZZZZZZZZZZZZZZZZZZSWITCH TO VAL MODEZZZZZZZZZZZZZZZZZZZZZ")
    model.vae.train(is_train)
    model.pgm.eval()
    model.predictor.eval()
    stats = {k: 0 for k in ["loss", "aux_loss", "elbo", "nll", "kl", "n"]}
    steps_skipped = 0

    dag_vars = list(model.pgm.variables.keys())
    print(dag_vars)
    if is_train and isinstance(optimizers, tuple):
        optimizer, lagrange_opt = optimizers
    else:
        preds = {k: [] for k in dag_vars}
        targets = {k: [] for k in dag_vars}
        # train_set = copy.deepcopy(dataloaders["train"].dataset.samples)

    loader = tqdm(
        enumerate(dataloaders[split]), total=len(dataloaders[split]), mininterval=0.1
    )
    from collections import deque

    # Initialize a deque with a fixed size of 100
    grad_norm_buffer = deque(maxlen=100)

    # Variables to keep track of the sum and average
    sum_grad_norm = 0.0
    avg_grad_norm = 0.0

    # for i, batch in loader:
    i, batch = next(iter(loader))
    bs = batch["x"].shape[0]
    batch = preprocess(batch)
    # print("batch",batch)
    # if i>5:
    #     break
    with torch.no_grad():
        # randomly intervene on a single parent do(pa_k ~ p(pa_k))
        do = {}
        # do_k =  random.choice(dag_vars)
        do_k = 'finding'
        do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
        if is_train:
            # print(do_k)
            do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
        else:
            # idx = torch.randperm(train_set[do_k].shape[0])
            # do[do_k] = train_set[do_k].clone()[idx][:bs]
            do[do_k] = batch[do_k].clone()[torch.randperm(bs)]
            do = preprocess(do)

    with torch.set_grad_enabled(is_train):
        if not is_train:
            args.cf_particles = 5 if i == 0 else 1
        # print("DO",do,"DO")
        out, pa, do, obs_x = model.forward(batch, do, elbo_fn, cf_particles=args.cf_particles)

        if torch.isnan(out["loss"]):
            model.zero_grad(set_to_none=True)
            steps_skipped += 1
            # continue

    if is_train:
        # print("==============================================")
        args.step = i + (args.epoch - 1) * len(dataloaders[split])
        optimizer.zero_grad(set_to_none=True)
        lagrange_opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if len(grad_norm_buffer) == 100:
            # Subtract the oldest value from the sum
            sum_grad_norm -= grad_norm_buffer[0]

        grad_norm_buffer.append(grad_norm)
        sum_grad_norm += grad_norm

        # Compute the running average
        avg_grad_norm = sum_grad_norm / len(grad_norm_buffer)
        if i<1000:
            if True:
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                model.lmbda.data.clamp_(min=0)
                ema.update()
            else:
                steps_skipped += 1
                print(f"Steps skipped: {steps_skipped} - grad_norm: {grad_norm:.3f}")
        else:
            if grad_norm< avg_grad_norm*0.6:
                optimizer.step()
                lagrange_opt.step()  # gradient ascent on lmbda
                model.lmbda.data.clamp_(min=0)
                ema.update()
            else:
                steps_skipped += 1
                print(f"Steps skipped: {steps_skipped} - grad_norm: {grad_norm:.3f}")
    else:  # evaluation
        # print("++++++++++++++++++++++++++++++++++++++++++++++")
        with torch.no_grad():
            preds_cf = ema.ema_model.predictor.predict(**out["cfs"])
            for k, v in preds_cf.items():
                preds[k].extend(v)
            # interventions are the targets for prediction
            for k in targets.keys():
                # print("k keys",k)
                t_k = do[k].clone() if k in do.keys() else out["cfs"][k].clone()
                # print("tk",t_k)
                # print("tgs",targets)
                # print("pcf",preds_cf)
                # targets[k].extend(inv_preprocess({k: t_k})[k])
                targets[k].extend({k: t_k}[k])

    if i % args.plot_freq == 0:
        save_path = os.path.join(args.save_dir, f'{args.step}_{split}_{do_k}_cfs.pdf')
        save_plot(save_path, batch, out['cfs'], do, out['var_cf_x'], num_images=3,step = i,is_train=is_train,epoch = epoch)

    stats["n"] += bs
    stats["loss"] += out["loss"].item() * bs
    stats["aux_loss"] += out["aux_loss"].item() * args.alpha * bs
    stats["elbo"] += out["elbo"] * bs
    stats["nll"] += out["nll"] * bs
    stats["kl"] += out["kl"] * bs
    stats = update_stats(stats, elbo_fn)  # aux_model stats
    loader.set_description(
        f"[{split}] lmbda: {model.lmbda.data.item():.3f}, "
        + f", ".join(
            f'{k}: {v / stats["n"]:.3f}' for k, v in stats.items() if k != "n"
        )
        + (f", grad_norm: {grad_norm:.3f}" if is_train else "")
    )
        
        
    # Ensure the unused variables are deleted and memory is freed
    # del loader, batch, out, do, do_k

    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()  # Collect garbage
    
    stats = {k: v / stats["n"] for k, v in stats.items() if k != "n"}
    return (stats, out, pa, do, obs_x) if is_train else (stats, get_metrics(args.dataset, preds, targets))


import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", help="experiment name.", type=str, default="123")
parser.add_argument(
    "--data_dir", help="data directory to load form.", type=str, default="/home/yifei/Documents/MIMIC_data/60k_clean"
)
parser.add_argument(
    "--load_path", help="Path to load checkpoint.", type=str, default=""
)
parser.add_argument(
    "--pgm_path",
    help="path to load pgm checkpoint.",
    type=str,
    default="../../checkpoints/a_r_s_f/pgm-clean_uncer_30%_P/checkpoint.pt",
)
parser.add_argument(
    "--predictor_path",
    help="path to load predictor checkpoint.",
    type=str,
    default="../../checkpoints/a_r_s_f/aux_mimic-clean_uncer_30%_P/checkpoint.pt",
)
parser.add_argument(
    "--vae_path",
    help="path to load vae checkpoint.",
    type=str,
    default="../../checkpoints/a_r_s_f/hvae_clean_uncer_30%_P/checkpoint.pt",
)
parser.add_argument("--seed", help="random seed.", type=int, default=7)
parser.add_argument(
    "--deterministic",
    help="toggle cudNN determinism.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--testing", help="test model.", action="store_true", default=False
)
# training
parser.add_argument("--epochs", help="num training epochs.", type=int, default=1)
parser.add_argument("--bs", help="batch size.", type=int, default=8)
parser.add_argument("--lr", help="learning rate.", type=float, default=1e-4)
parser.add_argument(
    "--lr_lagrange", help="learning rate for multipler.", type=float, default=1e-2
)
parser.add_argument(
    "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
)
parser.add_argument("--alpha", help="aux loss multiplier.", type=float, default=1)
parser.add_argument(
    "--lmbda_init", help="lagrange multiplier init.", type=float, default=0
)
parser.add_argument(
    "--damping", help="lagrange damping scalar.", type=float, default=100
)
parser.add_argument("--do_pa", help="intervened parent.", type=str, default=None)
parser.add_argument("--eval_freq", help="epochs per eval.", type=int, default=1)
parser.add_argument("--plot_freq", help="steps per plot.", type=int, default=500)
parser.add_argument("--imgs_plot", help="num images to plot.", type=int, default=10)
parser.add_argument(
    "--cf_particles", help="num counterfactual samples.", type=int, default=1
)
args = parser.parse_known_args()[0]

# update hparams if loading checkpoint
if args.load_path:
    if os.path.isfile(args.load_path):
        print(f"\nLoading checkpoint: {args.load_path}")
        ckpt = torch.load(args.load_path)
        ckpt_args = {k: v for k, v in ckpt["hparams"].items() if k != "load_path"}
        if args.data_dir is not None:
            ckpt_args["data_dir"] = args.data_dir
        if args.testing:
            ckpt_args["testing"] = args.testing
        vars(args).update(ckpt_args)
    else:
        print(f"Checkpoint not found at: {args.load_path}")

seed_all(args.seed, args.deterministic)

# Load predictors
print(f"\nLoading predictor checkpoint: {args.predictor_path}")
predictor_checkpoint = torch.load(args.predictor_path)
predictor_args = Hparams()
predictor_args.update(predictor_checkpoint["hparams"])
predictor = ChestPGM(predictor_args).cuda()
predictor.load_state_dict(predictor_checkpoint["ema_model_state_dict"])

# # for backwards compatibility
# if not hasattr(predictor_args, "dataset"):
#     predictor_args.dataset = "ukbb"
# if hasattr(predictor_args, "loss_norm"):
#     args.loss_norm


# if args.data_dir != "":
#     predictor_args.data_dir = args.data_dir
# dataloaders = setup_dataloaders(predictor_args)
# elbo_fn = TraceStorage_ELBO(num_particles=1)

# test_stats = sup_epoch(
#     predictor_args,
#     predictor,
#     None,
#     dataloaders["test"],
#     elbo_fn,
#     optimizer=None,
#     is_train=False,
# )
# stats = eval_epoch(predictor_args, predictor, dataloaders["test"])
# print("test | " + " - ".join(f"{k}: {v:.4f}" for k, v in stats.items()))

# Load PGM
print(f"\nLoading PGM checkpoint: {args.pgm_path}")
pgm_checkpoint = torch.load(args.pgm_path)
pgm_args = Hparams()
pgm_args.update(pgm_checkpoint["hparams"])
pgm = ChestPGM(pgm_args).cuda()
pgm.load_state_dict(pgm_checkpoint["ema_model_state_dict"])

# # for backwards compatibility
# if not hasattr(pgm_args, "dataset"):
#     pgm_args.dataset = "ukbb"
# if args.data_dir != "":
#     pgm_args.data_dir = args.data_dir
# dataloaders = setup_dataloaders(pgm_args)
# elbo_fn = TraceStorage_ELBO(num_particles=1)

# test_stats = sup_epoch(
#     pgm_args, pgm, None, dataloaders["test"], elbo_fn, is_train=False
# )

# Load deep VAE
print(f"\nLoading VAE checkpoint: {args.vae_path}")
vae_checkpoint = torch.load(args.vae_path)
vae_args = Hparams()
# vae_args.update({'dataset':args.dataset})
vae_args.update(vae_checkpoint["hparams"])
print(vae_checkpoint['hparams'])
if not hasattr(vae_args, "cond_prior"):  # for backwards compatibility
    vae_args.cond_prior = False
# vae_args.kl_free_bits = vae_args.free_bits
vae = HVAE(vae_args).cuda()
vae.load_state_dict(vae_checkpoint["ema_model_state_dict"])

# setup current experiment args
args.beta = vae_args.beta
args.parents_x = vae_args.parents_x
args.input_res = vae_args.input_res
args.grad_clip = vae_args.grad_clip
args.grad_skip = vae_args.grad_skip
args.elbo_constraint = 1.841216802597046  # train set elbo constraint
args.wd = vae_args.wd
args.betas = vae_args.betas

# init model
if not hasattr(vae_args, "dataset"):
    args.dataset = "mimic"
model = DSCM(args, pgm, predictor, vae)
ema = EMA(model, beta=args.ema_rate)
model.cuda()
ema.cuda()

# setup data
pgm_args.concat_pa = False
pgm_args.bs = args.bs
from train_pgm import setup_dataloaders

dataloaders = setup_dataloaders(pgm_args)

# Train model
if not args.testing:
    args.save_dir = setup_directories(args, ckpt_dir="../../checkpoints")
    writer = setup_tensorboard(args, model)
    logger = setup_logging(args)
    writer.add_custom_scalars(
        {
            "loss": {"loss": ["Multiline", ["loss/train", "loss/valid"]]},
            "aux_loss": {
                "aux_loss": ["Multiline", ["aux_loss/train", "aux_loss/valid"]]
            },
        }
    )

    # setup loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=1)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if n != "lmbda"],
        lr=args.lr,
        weight_decay=args.wd,
        betas=args.betas,
    )
    lagrange_opt = torch.optim.AdamW(
        [model.lmbda],
        lr=args.lr_lagrange,
        betas=args.betas,
        weight_decay=0,
        maximize=True,
    )
    optimizers = (optimizer, lagrange_opt)

    # load checkpoint
    if args.load_path:
        if os.path.isfile(args.load_path):
            args.start_epoch = ckpt["epoch"]
            args.step = ckpt["step"]
            args.best_loss = ckpt["best_loss"]
            model.load_state_dict(ckpt["model_state_dict"])
            ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            lagrange_opt.load_state_dict(ckpt["lagrange_opt_state_dict"])
        else:
            print("Checkpoint not found: {}".format(args.load_path))
    else:
        args.start_epoch, args.step = 0, 0
        # args.best_loss = float("inf")
        args.best_loss = float(0)

    for k in sorted(vars(args)):
        logger.info(f"--{k}={vars(args)[k]}")

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch + 1
        logger.info(f"Epoch: {args.epoch}")
        # print("[[[[[[[[[[[[[[[[[[[[[[[[TRAIN EPOCH]]]]]]]]]]]]]]]]]]]]]]]]")
        stats, out, pa, do, obs_x = cf_epoch(
            args, model, ema, dataloaders, elbo_fn, optimizers, split="train", epoch=epoch
        )
        
        
        
        
        
        
        
cfs = out['cfs']
obs_imgs = obs_x.cpu().detach().numpy()       
cf_imgs = cfs['x'].cpu().detach().numpy()     

import torch.nn.functional as F
import torchvision, torchvision.transforms
import torchxrayvision as xrv
import numpy as np

xrv.datasets.XRayResizer.ENGINE = 'cv2'
  
weights = 'densenet121-res224-all'       
model = xrv.models.get_model(weights)        
        
# for i in range(8):
    
def txv_getlabel(img, model, feats=False, cuda= True, resize=True): 
    if cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') 
        
    img_obs = xrv.datasets.normalize(((img[0,:,:]+1)*127.5).astype(np.uint8), 255)       
    
    # Check that images are 2D arrays
    if len(img_obs.shape) > 2:
        img_obs = img_obs[0,:, :]
    if len(img_obs.shape) < 2:
        print("error, dimension lower than 2 for image")       
    # Add color channel
    img_obs = img_obs[None, :, :]
    # the models will resize the input to the correct size so this is optional.
    if resize:
        # print(f"Global setting: XRayResizer engine set to: {xrv.datasets.XRayResizer.ENGINE}")

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(224)])
    else:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img_obs = transform(img_obs)
    
    model = xrv.models.get_model(weights)
    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img_obs).unsqueeze(0)
        if cuda:
            img = img.to(device)
            model = model.to(device)
            
        if feats:
            feats = model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

        preds = model(img).cpu()
        output = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy())) 
    return output
        
loss = 0
for i in range(8):        
    # i =1 
    out_obs = txv_getlabel(obs_imgs[i,:,:,:], model)
    out_cf =  txv_getlabel(cf_imgs[i,:,:,:], model)       
    
    FIND_CAT = ["no disease", "Effusion", "Pneumonia", "Consolidation", "Lung Opacity"] 
    
    cf_disease = FIND_CAT[do["finding"][i,:].cpu().argmax(-1)]
    obs_disease = FIND_CAT[pa["finding"][i,:].cpu().clone().squeeze().numpy().argmax(-1)]        
            
    disease = cf_disease if cf_disease != "no disease" else (obs_disease if obs_disease != "no disease" else "no finding")
    if disease =="no finding":
        loss += 100 * abs(sum([out_obs[key] for key in FIND_CAT[1:]])/4 - sum([out_cf[key] for key in FIND_CAT[1:]])/4)
    elif disease == "Pneumonia":
        loss += 100 * abs(sum([out_obs[key] for key in FIND_CAT[2:]])/3 - sum([out_cf[key] for key in FIND_CAT[2:]])/3)
    elif disease == "Effusion":
        loss += 100*abs(out_obs[disease] - out_cf[disease])
    print(loss)
loss /= 8                
         
        
        
        





# 假设 img 是一个 PyTorch 张量
img = torch.rand((1, 256, 256))  # 示例张量

# 将 img 转换为 uint8 格式
img_uint8 = ((img[0, :, :] + 1) * 127.5).clamp(0, 255).to(torch.uint8)

# 将 img_uint8 转换为 NumPy 数组
img_uint8_np = img_uint8.cpu().numpy()

# 调用 xrv.datasets.normalize
normalized_img = xrv.datasets.normalize(img_uint8_np, 255)
print(normalized_img)























        
        