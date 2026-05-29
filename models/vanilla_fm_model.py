"""
Vanilla conditional flow matching in the pix2pix framework.

Integrates mentor reference scripts:
  - flow_matching.py:     MONAI UNet, x1 L1 + perceptual, logit-normal t, tanh, Heun ODE via x1->v
  - flow_matching_v.py:   MONAI UNet, velocity MSE, uniform t, Heun ODE on raw v

Default backbone: MONAI DiffusionModelUNet (~11M params with 64,128,192 + res=2 + attn 0,0,1).
Legacy: --fm_backbone custom (skip U-Net, spatial t channel).
"""
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .fm_perceptual import build_fm_perceptual
from .mentor_flow_net import MentorFlowNet


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.conv(x)
        return self.res(x)


class _UpSkip(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, n_res: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.res(x)


class _CondUNet(nn.Module):
    """Legacy custom U-Net (spatial t channel). Use --fm_backbone monai for mentor parity."""

    def __init__(self, in_ch: int = 7, out_ch: int = 3,
                 channels: Tuple[int, int, int] = (32, 64, 96),
                 num_res_blocks: int = 1):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.res1 = nn.Sequential(*[_ResBlock(c1) for _ in range(num_res_blocks)])
        self.down1 = _Down(c1, c2, num_res_blocks)
        self.down2 = _Down(c2, c3, num_res_blocks)
        self.mid = nn.Sequential(*[_ResBlock(c3) for _ in range(max(1, num_res_blocks))])
        self.up2 = _UpSkip(c3, c2, c2, num_res_blocks)
        self.up1 = _UpSkip(c2, c1, c1, num_res_blocks)
        self.head = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x):
        h1 = self.res1(self.stem(x))
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h = self.mid(h3)
        h = self.up2(h, h2)
        h = self.up1(h, h1)
        return self.head(h)


class VanillaFMModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="aligned", no_dropout=True)
        parser.add_argument("--fm_backbone", type=str, default="monai",
                            choices=["monai", "custom"],
                            help="monai=mentor DiffusionModelUNet; custom=legacy skip U-Net")
        parser.add_argument("--fm_loss", type=str, default="x1",
                            choices=["x1", "velocity"],
                            help="x1=flow_matching.py; velocity=flow_matching_v.py")
        parser.add_argument("--fm_channels", type=str, default="64,128,192",
                            help="UNet channels (monai or custom)")
        parser.add_argument("--fm_attn_levels", type=str, default="0,0,0",
                            help="MONAI attention per level; use 0,0,0 at 1024² (OOM if 0,0,1)")
        parser.add_argument("--fm_num_head_channels", type=int, default=32,
                            help="MONAI attention head channels (monai only)")
        parser.add_argument("--fm_num_res_blocks", type=int, default=2,
                            help="ResBlocks per UNet level")
        parser.add_argument("--fm_use_tanh", action="store_true",
                            help="tanh on net output (flow_matching.py; x1 + monai)")
        parser.add_argument("--fm_time_dist", type=str, default="logit_normal",
                            choices=["uniform", "logit_normal"],
                            help="t sampling: logit_normal (x1 mentor) or uniform (velocity)")
        parser.add_argument("--fm_P_mean", type=float, default=-0.8,
                            help="Logit-normal t: mean (flow_matching.py)")
        parser.add_argument("--fm_P_std", type=float, default=0.8,
                            help="Logit-normal t: std")
        parser.add_argument("--fm_steps", type=int, default=25,
                            help="ODE steps at test")
        parser.add_argument("--fm_val_steps", type=int, default=8,
                            help="ODE steps during training validation")
        parser.add_argument("--fm_sample_method", type=str, default="heun",
                            choices=["euler", "heun"],
                            help="ODE solver at val / test")
        if is_train:
            parser.add_argument("--fm_lambda_perc", type=float, default=0.1,
                                help="Perceptual weight (x1 mode; flow_matching.py; 0=off)")
            parser.add_argument("--fm_perc_size", type=int, default=256,
                                help="Downsample for perceptual at 1024² (0=full res; 256 saves VRAM)")
            parser.add_argument("--fm_lambda_vel", type=float, default=0.0,
                                help="Aux velocity MSE in x1 mode (0=off)")
            parser.add_argument("--fm_lambda_l1", type=float, default=0.0,
                                help="Extra L1 on x1_hat (usually 0)")
            parser.add_argument("--fm_lambda_sample_l1", type=float, default=0.0,
                                help="Optional ODE sample L1 in train loop")
            parser.add_argument("--fm_train_sample_steps", type=int, default=0,
                                help="ODE steps when fm_lambda_sample_l1>0 (0=fm_steps)")
            parser.add_argument("--fm_sample_l1_prob", type=float, default=1.0,
                                help="Fraction of iters with ODE sample L1")
            parser.add_argument("--fm_train_sample_method", type=str, default="euler",
                                choices=["euler", "heun"],
                                help="ODE solver for train sample L1")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.fm_loss_mode = str(getattr(opt, "fm_loss", "x1"))
        self.use_monai = str(getattr(opt, "fm_backbone", "monai")) == "monai"

        self.loss_names = ["FM"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]

        ch = tuple(int(x) for x in opt.fm_channels.split(","))
        n_res = int(opt.fm_num_res_blocks)
        use_tanh = bool(getattr(opt, "fm_use_tanh", False)) and self.fm_loss_mode == "x1"

        if self.use_monai:
            attn = tuple(bool(int(a)) for a in opt.fm_attn_levels.split(","))
            self.netG: Union[MentorFlowNet, _CondUNet] = MentorFlowNet(
                in_ch=opt.input_nc,
                out_ch=opt.output_nc,
                channels=ch,
                attention_levels=attn,
                num_res_blocks=n_res,
                num_head_channels=int(opt.fm_num_head_channels),
                use_tanh=use_tanh,
            )
        else:
            self.netG = _CondUNet(
                in_ch=opt.input_nc + opt.output_nc + 1,
                out_ch=opt.output_nc,
                channels=ch,  # type: ignore[arg-type]
                num_res_blocks=n_res,
            )

        lam_perc = float(getattr(opt, "fm_lambda_perc", 0.0)) if self.isTrain else 0.0
        perc_size = int(getattr(opt, "fm_perc_size", 256))
        self.perceptual_loss_fn = None
        self._perc_backend = "off"
        if self.isTrain and self.fm_loss_mode == "x1" and lam_perc > 0:
            self.perceptual_loss_fn, self._perc_backend = build_fm_perceptual(
                self.device, lam_perc, perc_size=perc_size,
            )
            if self.perceptual_loss_fn is None:
                print(
                    "Warning: no perceptual backend (monai/lpips/torchvision); "
                    "install lpips or fix monai — fm_lambda_perc ignored"
                )
            else:
                print(
                    f"FM perceptual: backend={self._perc_backend} "
                    f"lambda={lam_perc} downsample={perc_size if perc_size > 0 else 'full'}"
                )

        if self.isTrain:
            if self.perceptual_loss_fn is not None:
                self.loss_names.append("Perc")
            if float(getattr(opt, "fm_lambda_vel", 0.0)) > 0:
                self.loss_names.append("Vel")
            if float(getattr(opt, "fm_lambda_l1", 0.0)) > 0:
                self.loss_names.append("L1")
            if float(getattr(opt, "fm_lambda_sample_l1", 0.0)) > 0:
                self.loss_names.append("L1s")

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.netG.to(self.device)
            if len(self.gpu_ids) > 1:
                self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dist = str(getattr(self.opt, "fm_time_dist", "logit_normal"))
        if self.fm_loss_mode == "velocity":
            return torch.rand(batch_size, device=device)
        if dist == "logit_normal":
            p_mean = float(getattr(self.opt, "fm_P_mean", -0.8))
            p_std = float(getattr(self.opt, "fm_P_std", 0.8))
            return torch.sigmoid(p_mean + p_std * torch.randn(batch_size, device=device))
        return torch.rand(batch_size, device=device)

    def _net_forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.use_monai:
            return self.netG(x_t, t, cond)
        B, _, H, W = x_t.shape
        t_ch = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([x_t, cond, t_ch], dim=1)
        return self.netG(inp)

    def _pred_x1(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._net_forward(x_t, t, self.real_A)

    def _velocity_from_x1(self, x_t: torch.Tensor, t: torch.Tensor,
                          x1_hat: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1_hat = x1_hat if x1_hat is not None else self._pred_x1(x_t, t)
        denom = (1.0 - t).view(-1, 1, 1, 1).clamp(min=1e-5)
        return (x1_hat - x_t) / denom

    def _velocity_raw(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._net_forward(x_t, t, self.real_A)

    def _ode_step_update(self, x: torch.Tensor, t0: torch.Tensor, t1: torch.Tensor,
                         dt: torch.Tensor, method: str, is_last: bool) -> torch.Tensor:
        B = x.shape[0]
        if self.fm_loss_mode == "velocity":
            v0 = self._velocity_raw(x, t0)
            if method == "euler" or is_last:
                return x + dt * v0
            x_mid = x + dt * v0
            v1 = self._velocity_raw(x_mid, t1)
            return x + dt * (v0 + v1) * 0.5

        x1_0 = self._pred_x1(x, t0)
        v0 = self._velocity_from_x1(x, t0, x1_0)
        if method == "euler" or is_last:
            return x + dt * v0
        x_mid = x + dt * v0
        x1_1 = self._pred_x1(x_mid, t1)
        v1 = self._velocity_from_x1(x_mid, t1, x1_1)
        return x + dt * (v0 + v1) * 0.5

    def _integrate(self, cond_A: torch.Tensor, x: torch.Tensor, steps: int,
                   method: str) -> torch.Tensor:
        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            x = self._ode_step_update(x, t0, t1, dt, method, is_last=(i == steps - 1))

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def _integrate_train_ode(self, cond_A, x, steps, method):
        from torch.utils.checkpoint import checkpoint

        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        B = cond_A.shape[0]
        ts = torch.linspace(0.0, 1.0, steps + 1, device=cond_A.device, dtype=cond_A.dtype)

        def step_ckpt(x_in, t0, t1, dt, is_last):
            return self._ode_step_update(x_in, t0, t1, dt, method, is_last)

        for i in range(steps):
            t0 = ts[i].expand(B)
            t1 = ts[i + 1].expand(B)
            dt = ts[i + 1] - ts[i]
            is_last = i == steps - 1
            if x.requires_grad:
                x = checkpoint(step_ckpt, x, t0, t1, dt, is_last, use_reentrant=False)
            else:
                x = step_ckpt(x, t0, t1, dt, is_last)

        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

    def _ode_config_test(self):
        return int(self.opt.fm_steps), str(getattr(self.opt, "fm_sample_method", "heun"))

    def forward(self):
        steps, method = self._ode_config_test()
        self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def compute_visuals(self):
        with torch.no_grad():
            steps, method = self._ode_config_test()
            self.fake_B = self.sample(self.real_A, steps=steps, method=method)

    def _loss_fm(self):
        x1 = self.real_B
        x0 = torch.randn_like(x1)
        t = self._sample_t(x1.shape[0], x1.device)
        t_bc = t.view(-1, 1, 1, 1)
        xt = (1.0 - t_bc) * x0 + t_bc * x1

        if self.fm_loss_mode == "velocity":
            v_tgt = x1 - x0
            v_pred = self._velocity_raw(xt, t)
            self.loss_FM = F.mse_loss(v_pred, v_tgt)
            loss = self.loss_FM
            lam_perc = float(getattr(self.opt, "fm_lambda_perc", 0.0))
            if self.perceptual_loss_fn is not None:
                x1_hat = xt + (1.0 - t_bc) * v_pred
                self.loss_Perc = self.perceptual_loss_fn(x1_hat, x1) * lam_perc
                loss = loss + self.loss_Perc
        else:
            x1_hat = self._pred_x1(xt, t)
            self.loss_FM = F.l1_loss(x1_hat, x1)
            loss = self.loss_FM
            lam_perc = float(getattr(self.opt, "fm_lambda_perc", 0.0))
            if self.perceptual_loss_fn is not None:
                self.loss_Perc = self.perceptual_loss_fn(x1_hat, x1) * lam_perc
                loss = loss + self.loss_Perc

        lam_vel = float(getattr(self.opt, "fm_lambda_vel", 0.0))
        if lam_vel > 0 and self.fm_loss_mode == "x1":
            v_star = x1 - x0
            v_pred = self._velocity_from_x1(xt, t)
            self.loss_Vel = F.mse_loss(v_pred, v_star) * lam_vel
            loss = loss + self.loss_Vel

        lam_path = float(getattr(self.opt, "fm_lambda_l1", 0.0))
        if lam_path > 0 and self.fm_loss_mode == "x1":
            self.loss_L1 = F.l1_loss(x1_hat, x1) * lam_path
            loss = loss + self.loss_L1

        # Ensure loggable scalars exist for every registered loss name.
        for name in self.loss_names:
            attr = "loss_" + name
            if not hasattr(self, attr):
                setattr(self, attr, loss.new_tensor(0.0))
        return loss

    def _loss_ode_sample_l1(self):
        x1 = self.real_B
        override = int(getattr(self.opt, "fm_train_sample_steps", 0))
        steps = override if override > 0 else int(self.opt.fm_steps)
        method = str(getattr(self.opt, "fm_train_sample_method", "euler"))
        fake = self._integrate_train_ode(self.real_A, torch.randn_like(x1), steps, method)
        lam = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        self.loss_L1s = F.l1_loss(fake, x1) * lam
        return self.loss_L1s

    def backward_G(self):
        self._loss_fm().backward()
        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        if lam_sample > 0 and random.random() < float(getattr(self.opt, "fm_sample_l1_prob", 1.0)):
            self._loss_ode_sample_l1().backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad(set_to_none=True)
        self._loss_fm().backward()
        lam_sample = float(getattr(self.opt, "fm_lambda_sample_l1", 0.0))
        if lam_sample > 0 and random.random() < float(getattr(self.opt, "fm_sample_l1_prob", 1.0)):
            self._loss_ode_sample_l1().backward()
        self.optimizer_G.step()

    @torch.no_grad()
    def sample(self, cond_A: torch.Tensor, steps: Optional[int] = None,
               method: Optional[str] = None) -> torch.Tensor:
        default_steps, default_method = self._ode_config_test()
        steps = int(steps if steps is not None else default_steps)
        method = method or default_method
        x_init = torch.randn(
            cond_A.shape[0], self.opt.output_nc, cond_A.shape[2], cond_A.shape[3],
            device=cond_A.device, dtype=cond_A.dtype,
        )
        return self._integrate(cond_A, x_init, steps, method)
