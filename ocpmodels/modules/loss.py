import torch
from torch import nn

from ocpmodels.common import distutils


class L2MAELoss(nn.Module):
	def __init__(self, reduction="mean"):
		super().__init__()
		self.reduction = reduction
		assert reduction in ["mean", "sum"]

	def forward(self, input: torch.Tensor, target: torch.Tensor):
		dists = torch.norm(input - target, p=2, dim=-1)
		if self.reduction == "mean":
			return torch.mean(dists)
		elif self.reduction == "sum":
			return torch.sum(dists)


class AtomwiseL2Loss(nn.Module):
	def __init__(self, reduction="mean"):
		super().__init__()
		self.reduction = reduction
		assert reduction in ["mean", "sum"]

	def forward(
		self,
		input: torch.Tensor,
		target: torch.Tensor,
		natoms: torch.Tensor,
	):
		assert natoms.shape[0] == input.shape[0] == target.shape[0]
		assert len(natoms.shape) == 1  # (nAtoms, )

		dists = torch.norm(input - target, p=2, dim=-1)
		loss = natoms * dists

		if self.reduction == "mean":
			return torch.mean(loss)
		elif self.reduction == "sum":
			return torch.sum(loss)


class DDPLoss(nn.Module):
	def __init__(self, loss_fn, reduction="mean"):
		super().__init__()
		self.loss_fn = loss_fn
		self.loss_fn.reduction = "sum"
		self.reduction = reduction
		assert reduction in ["mean", "sum"]

	def forward(
		self,
		input: torch.Tensor,
		target: torch.Tensor,
		natoms: torch.Tensor = None,
		batch_size: int = None,
	):
		if natoms is None:
			loss = self.loss_fn(input, target)
		else:  # atom-wise loss
			loss = self.loss_fn(input, target, natoms)
		if self.reduction == "mean":
			num_samples = (
				batch_size if batch_size is not None else input.shape[0]
			)
			num_samples = distutils.all_reduce(
				num_samples, device=input.device
			)
			# Multiply by world size since gradients are averaged
			# across DDP replicas
			return loss * distutils.get_world_size() / num_samples
		else:
			return loss

class EvidentialLoss(nn.Module):
	def __init__(self, lambda_="mean"):
		super().__init__()
		self.lambda_ = lambda_

	def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
		twoBlambda = 2*beta*(1+v)

		nll = 0.5*torch.log(np.pi/v)  \
			- alpha*torch.log(twoBlambda)  \
			+ (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
			+ torch.lgamma(alpha)  \
			- torch.lgamma(alpha+0.5)

		return torch.mean(nll) if reduce else nll

	def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
		KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
			+ 0.5*v2/v1  \
			- 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
			- 0.5 + a2*torch.log(b1/b2)  \
			- (torch.lgamma(a1) - torch.lgamma(a2))  \
			+ (a1 - a2)*torch.digamma(a1)  \
			- (b1 - b2)*a1/b1
		return KL

	def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
		error = torch.abs(y-gamma)

		if kl:
			kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
			reg = error*kl
		else:
			evi = 2*v+(alpha)
			reg = error*evi

		return torch.mean(reg) if reduce else reg

	def EvidentialRegression(y_true, evidential_output, coeff):
		gamma, v, alpha, beta = torch.split(evidential_output, 1, dim=-1)
		loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
		loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
		return loss_nll + coeff * loss_reg

	def forward(self, input: torch.Tensor, target: torch.Tensor):
		return EvidentialRegression(input, target, coeff=self.lambda_)