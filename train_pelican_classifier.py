import logging
import os
import sys

from src.trainer import which
if which('nvidia-smi') is not None:
    min=8000
    deviceid = 0
    name, mem = os.popen('"nvidia-smi" --query-gpu=gpu_name,memory.total --format=csv,nounits,noheader').read().split('\n')[deviceid].split(',')
    print(mem)
    mem = int(mem)
    if mem < min:
        print('Less GPU memory than requested. Terminating.')
        sys.exit()

logger = logging.getLogger('')

import torch
from torch.utils.data import DataLoader

from src.models import PELICANClassifier
from src.models import tests
from src.trainer import Trainer
from src.trainer import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from src.trainer import init_optimizer, init_scheduler
from src.models.metrics_classifier import metrics, minibatch_metrics, minibatch_metrics_string


from src.dataloaders import initialize_datasets, collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000, sci_mode=False)


def main():

    # Initialize arguments -- Just
    args = init_argparse()

    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    if which('nvidia-smi') is not None:
        logger.info(f'Using {name} with {mem} MB of GPU memory')

    # Write input paramaters and paths to log
    logging_printout(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    if args.fix_data:
        torch.manual_seed(165937750084982)
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    if args.task.startswith('eval'):
        args.load = True

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
    model = PELICANClassifier(args.num_channels_m, args.num_channels1, args.num_channels2, args.num_channels_m_out,
                      activate_agg=args.activate_agg, activate_lin=args.activate_lin,
                      activation=args.activation, add_beams=args.add_beams, sig=args.sig, config1=args.config1, config2=args.config2, average_nobj=args.nobj_avg,
                      factorize=args.factorize, masked=args.masked, softmasked=args.softmasked,
                      activate_agg2=args.activate_agg2, activate_lin2=args.activate_lin2, mlp_out=args.mlp_out,
                      scale=args.scale, ir_safe=args.ir_safe, dropout = args.dropout, drop_rate=args.drop_rate, drop_rate_out=args.drop_rate_out, batchnorm=args.batchnorm,
                      device=device, dtype=dtype)
    
    model.to(device)

    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model, len(dataloaders['train']))
    scheduler, restart_epochs, summarize_csv, summarize = init_scheduler(args, optimizer)

    # Define a loss function.
    # loss_fn = torch.nn.functional.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        tests(model, dataloaders['train'], args, tests=['gpu'])

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, metrics, minibatch_metrics, minibatch_metrics_string, optimizer, scheduler, restart_epochs, summarize_csv, summarize, device, dtype)
    
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Set a CUDA variale that makes the results exactly reproducible on a GPU (on CPU they're reproducible regardless)
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # Train model.
    if not args.task.startswith('eval'):
        trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate(splits=['test'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    main()
