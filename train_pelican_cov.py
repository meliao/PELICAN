import torch
from torch.utils.data import DataLoader

import logging
import os

from src.models import PELICANRegression
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_regression import metrics, minibatch_metrics, minibatch_metrics_string
from src.models.lorentz_metric import normsq4, dot4

from src.dataloaders import initialize_datasets, collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def main():

    # Initialize arguments -- Just
    args = init_argparse()
   
    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Write input paramaters and paths to log
    logging_printout(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    if args.task.startswith('eval'):
        args.load = True
        args.num_epoch = 0

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    # Initialize model
    model = PELICANRegression(args.num_channels_m, args.num_channels1, args.num_channels2, args.num_channels_m_out,
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, sig=args.sig, config1=args.config1, config2=args.config2,
                      activate_agg2=args.activate_agg2, activate_lin2=args.activate_lin2, mlp_out=args.mlp_out,
                      scale=args.scale, ir_safe=args.ir_safe, dropout = args.dropout, batchnorm=args.batchnorm, layernorm=args.layernorm,
                      device=device, dtype=dtype)
    
    model.to(device)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs, summarize = init_scheduler(args, optimizer)

    # Define a loss function. This is the loss function whose gradients are actually computed. 
    # The loss function in engine.py is used only for the log and choosing best epoch!

    loss_fn_inv = lambda predict, targets:  normsq4(predict - targets).abs().mean()
    loss_fn_m2 = lambda predict, targets:  (normsq4(predict) - normsq4(targets)).abs().mean()
    loss_fn_3d = lambda predict, targets:  (predict[:,[1,2,3]] - targets[:,[1,2,3]]).norm(dim=-1).mean()
    loss_fn_4d = lambda predict, targets:  (predict-targets).pow(2).sum(-1).mean()
    loss_fn = lambda predict, targets: 0.1 * loss_fn_inv(predict,targets) + loss_fn_m2(predict,targets) #+ 0.001 * loss_fn_4d(predict, targets)
    # loss_fn = lambda predict, targets: loss_fn_m2(predict,targets) + loss_fn_4d(predict, targets)
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        raise NotImplementedError()

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, summarize, device, dtype)
    
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Set a CUDA variale that makes the results exactly reproducible on a GPU (on CPU they're reproducible regardless)
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate(splits=['test'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()