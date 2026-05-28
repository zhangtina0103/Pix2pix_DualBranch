import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


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


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.conv(x)
        return self.res(x)


class _CondUNet(nn.Module):
    """
    Minimal conditional UNet for (vanilla) flow matching.

    Input: concat([x_t (B,3,H,W), cond_A (B,3,H,W), t_channel (B,1,H,W)]) => 7ch
    Output: velocity field v_theta (B,3,H,W)
    """

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
        self.up2 = _Up(c3, c2, num_res_blocks)
        self.up1 = _Up(c2, c1, num_res_blocks)
        self.head = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x):
        h1 = self.res1(self.stem(x))
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h = self.mid(h3)
        h = self.up2(h)
        h = self.up1(h)
        return self.head(h)


class VanillaFMModel(BaseModel):
    """
    Vanilla conditional flow matching (rectified flow) in the pix2pix framework.

    Uses aligned paired data (A,B) and trains a conditional vector field v(x,t,A)
    that transports x0~N(0,1) to B.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode="aligned", no_dropout=True)
        parser.add_argument("--fm_channels", type=str, default="32,64,96",
                            help="UNet channels e.g. 32,64,96 (tune params)")
        parser.add_argument("--fm_num_res_blocks", type=int, default=1,
                            help="ResBlocks per level (tune params)")
        parser.add_argument("--fm_steps", type=int, default=25,
                            help="Euler steps for sampling during test")
        if is_train:
            parser.add_argument("--fm_lambda_l1", type=float, default=0.0,
                                help="Optional extra L1 on sampled output (0 disables)")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["FM"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.model_names = ["G"]  # we will save netG as 'G'

        ch = tuple(int(x) for x in opt.fm_channels.split(","))
        self.netG = _CondUNet(
            in_ch=opt.input_nc + opt.output_nc + 1,
            out_ch=opt.output_nc,
            channels=ch,
            num_res_blocks=int(opt.fm_num_res_blocks),
        )
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.netG.to(self.device)
            self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def _v(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0,1]
        B, _, H, W = x_t.shape
        t_ch = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        inp = torch.cat([x_t, self.real_A, t_ch], dim=1)
        return self.netG(inp)

    def forward(self):
        # for visualization/test we sample from noise -> x1
        self.fake_B = self.sample(self.real_A, steps=int(self.opt.fm_steps))

    def compute_visuals(self):
        """Training skips forward(); create fake_B for visdom/HTML when display_freq fires."""
        with torch.no_grad():
            steps = int(getattr(self.opt, "fm_steps", 25))
            self.fake_B = self.sample(self.real_A, steps=steps)

    def backward_G(self):
        """
        Rectified flow matching:
          x0 ~ N(0,1)
          xt = (1-t)*x0 + t*x1
          v* = x1 - x0
          loss = || v_theta(xt,t,A) - v* ||^2
        """
        x1 = self.real_B
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t).view(-1, 1, 1, 1) * x0 + t.view(-1, 1, 1, 1) * x1
        v_star = x1 - x0
        v_pred = self._v(xt, t)
        self.loss_FM = F.mse_loss(v_pred, v_star)
        self.loss_FM.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    @torch.no_grad()
    def sample(self, cond_A: torch.Tensor, steps: int = 25) -> torch.Tensor:
        """
        Euler integrate dx/dt = v_theta(x,t,A) from t=0 to 1.
        """
        # temporarily set for conditioning
        real_A_prev = getattr(self, "real_A", None)
        self.real_A = cond_A
        x = torch.randn(cond_A.shape[0], self.opt.output_nc, cond_A.shape[2], cond_A.shape[3], device=cond_A.device)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((cond_A.shape[0],), i * dt, device=cond_A.device)
            x = x + dt * self._v(x, t)
        if real_A_prev is None:
            delattr(self, "real_A")
        else:
            self.real_A = real_A_prev
        return x.clamp(-1, 1)

