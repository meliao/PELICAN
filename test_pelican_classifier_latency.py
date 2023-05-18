import logging
import os
import sys
from typing import Tuple
from timeit import default_timer

import numpy as np
import pandas as pd
from scipy import io
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



class TopTaggingDataset:
    def __init__(self, 
                 momentum_arr: np.ndarray, 
                 energy_arr: np.ndarray, 
                 label_arr: np.ndarray,
                 add_beam: bool=False) -> None:
        """_summary_

        Args:
            momentum_arr (np.ndarray): Has shape (n_samples, max_n_particles, 3). Is zero padded along the 
            2nd axis for samples with fewer than the max number of particles
            energy_arr (np.ndarray): Has shape (n_samples, max_n_particles). Is zero-padded along the 2nd axis
            for samples with fewer than the max number of particles.
            label_arr (np.ndarray): _description_
        """

        if add_beam:
            self.bool_beam_added = True
            self.n_samples = energy_arr.shape[0]
            self.max_n_particles = energy_arr.shape[1] + 2

            self.four_momentum_arr = np.empty((self.n_samples, self.max_n_particles, 4), dtype=np.float32)
            self.four_momentum_arr[:, :2] = np.array([[1., 0., 0., 1.],
                                                    [1., 0., 0., -1.]])
            self.four_momentum_arr[:, 2:, 0] = energy_arr
            self.four_momentum_arr[:, 2:, 1:] = momentum_arr

        else:
            self.bool_beam_added = False
            self.n_samples, self.max_n_particles = energy_arr.shape

            self.four_momentum_arr = np.empty((self.n_samples, self.max_n_particles, 4), dtype=np.float32)
            self.four_momentum_arr[:, :, 0] = energy_arr
            self.four_momentum_arr[:, :, 1:] = momentum_arr


        # self.momentum_arr = momentum_arr.copy()
        # self.energy_arr = energy_arr
        self.n_particles = np.sum(energy_arr != 0., axis=1) + 2 * int(add_beam)
        # self.n_samples, self.max_n_particles = energy_arr.shape
        self.label_arr = label_arr

        

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Returns momentum arr, energy arr, n_particles
        """
        momentum_i = self.four_momentum_arr[idx, :, 1:]
        energy_i = self.four_momentum_arr[idx, :, 0]
        n_particles_i = self.n_particles[idx]
        return (momentum_i, energy_i, n_particles_i)

    def _get_class_subset(self, cls_value: int) -> None:
        bool_arr = self.label_arr == cls_value
        
        momentum_subset = self.four_momentum_arr[bool_arr, :, 1:]
        energy_subset = self.four_momentum_arr[bool_arr, :, 0]
        label_subset = self.label_arr[bool_arr]
        
        return TopTaggingDataset(momentum_subset, energy_subset, label_subset)
        
    def get_positive_class(self) -> None:
        return self._get_class_subset(1)
        
    def get_negative_class(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_class_subset(0)

    def truncate(self, n: int, idxes: np.ndarray=None) -> None:
        if idxes is None:
            idxes = np.random.choice(np.arange(self.n_samples), n, replace=False)
        else:
            assert idxes.size == n
        momentum_out = self.four_momentum_arr[idxes, :, 1:]
        energy_out = self.four_momentum_arr[idxes, :, 0]
        labels_out = self.label_arr[idxes]

        return TopTaggingDataset(momentum_out, energy_out, labels_out)


    def get_four_momentum_arr(self, idx: int) -> np.ndarray:
        """_summary_

        Args:
            idx (int): Index of the collision event

        Returns:
            np.ndarray: Has shape (n_particles, 4) where the columns are 
                        [E, p_x, p_y, p_z]
        """
        return self.four_momentum_arr[idx, :self.n_particles[idx]]
        # m, e, n = self.__getitem__(idx)
        # out = np.empty((n, 4), np.float32)
        # out[:, 0] = e[:n]
        # out[:, 1:] = m[:n]
        # return out

    def get_four_momentum_arr_padded(self, idx: int) -> np.ndarray:
        return self.four_momentum_arr[idx]
        # m, e, n = self.__getitem__(idx)
        # out = np.empty((self.max_n_particles, 4), np.float32)
        # out[:, 0] = e
        # out[:, 1:] = m
        # return out

    @classmethod
    def from_hdf5_file(cls, fp: str, add_beam: bool=False) -> None:
        store = pd.HDFStore(fp)
        x = store.select("table")
        
        momentum_cols = [i for i in x.columns if i.startswith("P")]
        energy_cols = [i for i in x.columns if i.startswith("E")]

        momentum_arr = x[momentum_cols].values
        energy_arr = x[energy_cols].values
        
        n_samples, max_n_particles = energy_arr.shape
        
        reshaped_momentum_arr = momentum_arr.reshape((n_samples, max_n_particles, 3))
        label_arr = x['is_signal_new'].values

        # This copy operation is performed so that the array will become contiguous in memory. 
        reshaped_momentum_arr_out = reshaped_momentum_arr.copy(order='C')
        
        
        store.close()
        return cls(reshaped_momentum_arr_out, energy_arr, label_arr, add_beam=add_beam)


class TopTaggingBatcher(torch.utils.data.Dataset):
    def __init__(self, 
                 top_tagging_obj: TopTaggingDataset, 
                 batch_size: int,
                 permutation: bool=False) -> None:
        self.top_tagging_obj = top_tagging_obj
        self.batch_size = batch_size
        if permutation:
            self.perm = np.random.permutation(self.top_tagging_obj.n_samples)
        else:
            self.perm = np.arange(self.top_tagging_obj.n_samples)
            
        self.n_batches = int(np.ceil(self.top_tagging_obj.n_samples / self.batch_size))
        self.need_last_off_size_batch = self.n_batches != self.top_tagging_obj.n_samples // self.batch_size
        
    def __len__(self):
        return self.n_batches
    
    def __iter__(self):
        
        last_perm_idx = 0
        for i in range(self.n_batches - int(self.need_last_off_size_batch)):
            first_perm_idx = i * self.batch_size
            last_perm_idx = (i + 1) * self.batch_size
            batch_idxes = self.perm[first_perm_idx:last_perm_idx]
            
            # Find the max number of particles in the batch. This will be different from 
            # batch to batch, but the network will accept inputs of any length along the 2nd
            # dimension.
            max_n_particles = int(np.max(self.top_tagging_obj.n_particles[batch_idxes])) 
            out = self.top_tagging_obj.four_momentum_arr[batch_idxes, :max_n_particles]
            
            yield torch.from_numpy(out)
            
        if self.need_last_off_size_batch:
            batch_idxes = self.perm[last_perm_idx:]
            
            max_n_particles = int(np.max(self.top_tagging_obj.n_particles[batch_idxes]))       
            out = self.top_tagging_obj.four_momentum_arr[batch_idxes, :max_n_particles]
            
            yield torch.from_numpy(out)   


def measure_prediction_latency(dset: TopTaggingDataset,
                                model: torch.nn.Module,
                                n_loops: int,
                                device: torch.cuda.Device) -> np.ndarray:

    batcher = TopTaggingBatcher(dset, 1)

    out_arr = np.empty((2, len(dset) * n_loops), np.float32)

    for loop in range(n_loops):
        # logging.info("Beginning loop %i / %i", loop+1, n_loops)

        n_particle_lst = []
        time_lst = []
        for i, sample in enumerate(batcher):
            sample = sample.to(device)

            # I do not know what this does, but the network expects it as input.
            # s = sample.shape
            # scalars = torch.cat((torch.ones(s[0], 2), torch.zeros(s[0], s[1])), dim=1).to(data.device)

            t_0 = default_timer()
            pred = model.forward_latency(sample)
            t_1 = default_timer()
            n_particle_lst.append(sample.shape[-2])
            time_lst.append(t_1 - t_0)

        mean_time = np.mean(time_lst)
        max_time = np.max(time_lst)
        min_time = np.min(time_lst)

        logging.info("On loop %i/%i, Mean: %f, Max: %f, Min: %f", loop+1, n_loops, mean_time, max_time, min_time)

        # Save time_lst and n_particle_lst in the out_arr
        time_arr = np.array(time_lst)
        n_particle_arr = np.array(n_particle_lst)
        out_arr[0, loop * len(dset): (loop + 1) * len(dset)] = time_arr
        out_arr[1, loop * len(dset): (loop + 1) * len(dset)] = n_particle_arr

    return out_arr

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

    if args.test_latency:
        N_LOOPS = 3
        SAVE_DATA_FP = '/home/meliao/projects/lorentz_group_random_features/data/generated/2023-02-01_PELICAN_prediction_latency.mat'
        TRUNCATE_NUM = 10_000
        dset = TopTaggingDataset.from_hdf5_file('/home/meliao/projects/lorentz_group_random_features/data/top_tagging/val.h5')
        dset = dset.truncate(TRUNCATE_NUM)

    else:
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


    if args.test_latency:
        out_pred_latency = measure_prediction_latency(dset, model, N_LOOPS, device)

        out_data_dd = {
            'prediction_latency': out_pred_latency
        }
        logging.info("Saving results to %s", SAVE_DATA_FP)
        io.savemat(SAVE_DATA_FP, out_data_dd)

    else:
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
