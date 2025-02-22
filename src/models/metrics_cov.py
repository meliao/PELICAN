import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from .lorentz_metric import normsq4, dot4

def metrics(predict, targets, loss_fn, prefix, logger=None):
    """
    This generates metrics reported at the end of each epoch and during validation/testing, as well as the logstring printed to the logger.
    """    
    if len(targets.shape)==2:
        targets = targets.unsqueeze(1)
    loss = loss_fn(predict,targets).numpy()
    angle = AngleDeviation(predict, targets)
    phisigma = PhiSigma(predict, targets)
    pTsigma = pTSigma(predict, targets)
    massdelta = MassSigma(predict, targets)
    loss_inv = loss_fn_inv(predict, targets)
    loss_m = loss_fn_m(predict, targets)
    loss_m2 = loss_fn_m2(predict, targets)
    loss_3d = loss_fn_3d(predict, targets)
    loss_4d = loss_fn_4d(predict, targets)

    w = 1 + 8 * targets.shape[1]

    metrics = {'loss': loss, '∆Ψ': angle, '∆φ': phisigma, '∆pT': pTsigma, '∆m': massdelta, 'loss_inv': loss_inv, 'loss_m': loss_m, 'loss_m2': loss_m2, 'loss_3d': loss_3d, 'loss_4d': loss_4d}
    with np.printoptions(precision=4):
        string = f' L: {loss:10.4f}, ∆Ψ: {str(angle):>{w}}, ∆φ: {str(phisigma):>{w}}, ∆pT: {str(pTsigma):>{w}}, ∆m: {str(massdelta):>{w}}, loss_inv: {loss_inv:10.4f}, loss_m: {loss_m:10.4f}, loss_m2: {loss_m2:10.4f}, loss_3d: {loss_3d:10.4f}, loss_4d: {loss_4d:10.4f}'
    return metrics, string

def minibatch_metrics(predict, targets, loss):
    """
    This computes metrics for each minibatch (if verbose mode is used). The logstring is defined separately in minibatch_metrics_string.
    """    
    if len(targets.shape)==2:
        targets = targets.unsqueeze(1)
    angle = AngleDeviation(predict, targets)
    phisigma = PhiSigma(predict, targets)
    pTsigma = pTSigma(predict, targets)
    massdelta = MassSigma(predict, targets)

    return [loss, angle, phisigma, pTsigma, massdelta]

def minibatch_metrics_string(metrics):
    L, psi, phi, pT, m = metrics
    with np.printoptions(precision=4):
        string = f'   L: {L:<12.4f}, ∆Ψ: {str(psi):>25}, ∆φ: {str(phi):>25}, ∆pT: {str(pT):>25}, ∆m: {str(m):>25}'
    return string

def AngleDeviation(predict, targets):
    """
    Measures the (always positive) angle between any two 3D vectors and returns the 68% quantile over the batch
    """
    angles = Angle3D(predict[...,1:4], targets[...,1:4])
    return  torch.quantile(angles, 0.68, dim=0).numpy()

def PhiSigma(predict, targets):
    """
    Measures the oriented angle between any two 2D vectors and returns  half of the 68% interquantile range over the batch
    """
    angles = Angle2D(predict[...,1:3], targets[...,1:3])
    return  iqr(angles, dim=0)

def Angle2D(u, v):
    """
    Measures the oriented angle between any two 2D vectors (allows batches)
    """
    dots = (u * v).sum(dim=-1)
    j = torch.tensor([[0,1],[-1,0]], device=u.device, dtype=u.dtype)
    dets = (u * torch.einsum("ab,...b->...a", j, v)).sum(dim=-1)
    angles = torch.atan2(dets, dots)
    return angles

def Angle3D(u, v):
    """
    Measures the (always positive) angle between any two 3D vectors (allows batches)
    """
    aux1 = u.norm(dim=-1).unsqueeze(-1) * v
    aux2 = v.norm(dim=-1).unsqueeze(-1) * u
    angles = 2*torch.atan2((aux1 - aux2).norm(dim=-1), (aux1 + aux2).norm(dim=-1))
    return angles

def MassSigma(predict, targets):
    """
    half of the 68% interquantile range over of relative deviation in mass
    """
    rel = ((normsq4(predict).abs().sqrt()-normsq4(targets).abs().sqrt())/normsq4(targets).abs().sqrt())
    return iqr(rel, dim=0)  # mass relative error

def pTSigma(predict, targets):
    """
     half of the 68% interquantile range of relative deviation in pT
    """
    rel = ((predict[...,[1,2]].norm(dim=-1)-targets[...,[1,2]].norm(dim=-1))/targets[...,[1,2]].norm(dim=-1))
    return iqr(rel, dim=0)  # pT relative error

def loss_fn_inv(predict, targets):
    return (normsq4(predict - targets).abs()+1e-6).sqrt().mean().numpy()

def loss_fn_m(predict, targets):
    return (mass(predict) - mass(targets)).abs().mean().numpy()

def loss_fn_m2(predict, targets):
    return (normsq4(predict) - normsq4(targets)).abs().mean().numpy()

def loss_fn_3d(predict, targets):
    return ((predict[...,1:4] - targets[...,1:4]).norm(dim=-1)).mean().numpy()

def loss_fn_4d(predict, targets):
    return (predict-targets).norm(dim=-1).mean().numpy()

def mass(x):
    norm=normsq4(x)
    return norm.sign() * norm.abs().sqrt()


def iqr(x, rng=(0.16, 0.84), dim=0):
    rng = sorted(rng)
    return ((torch.quantile(x,rng[1], dim=dim) - torch.quantile(x,rng[0],dim=dim)) * 0.5).numpy()