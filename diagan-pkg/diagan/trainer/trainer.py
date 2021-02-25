import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch_mimicry as mmc
from torch_mimicry.training import logger, metric_log

from diagan.trainer.scheduler import DRS_LRScheduler
from diagan.utils.plot import plot_gaussian_samples


class LogTrainer(mmc.training.Trainer):
    def __init__(
        self,
        output_path,
        netD,
        netG,
        optD,
        optG,
        dataloader,
        num_steps,
        netD_drs=None,
        optD_drs=None,
        dataloader_drs=None,
        netD_drs_ckpt_file=None,
        log_dir='./log',
        n_dis=1,
        lr_decay=None,
        device=None,
        netG_ckpt_file=None,
        netD_ckpt_file=None,
        print_steps=1,
        vis_steps=500,
        log_steps=50,
        save_steps=5000,
        flush_secs=30,
        logit_save_steps=500,
        amp=False,
        save_logits=True,
        topk=False,
        gold=False,
        gold_step=None,
        save_logit_after=0,
        stop_save_logit_after=100000,
        save_eval_logits=True,
    ):
        self.output_path = output_path
        self.logit_save_steps = logit_save_steps

        self.netD = netD
        self.netG = netG
        self.optD = optD
        self.optG = optG
        self.n_dis = n_dis
        self.lr_decay = lr_decay
        self.dataloader = dataloader
        self.num_steps = num_steps
        self.device = device
        self.log_dir = log_dir
        self.netG_ckpt_file = netG_ckpt_file
        self.netD_ckpt_file = netD_ckpt_file
        self.print_steps = print_steps
        self.vis_steps = vis_steps
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.amp = amp
        self.save_logits = save_logits
        self.save_logit_after = save_logit_after
        self.stop_save_logit_after = stop_save_logit_after
        self.save_eval_logits = save_eval_logits

        # for DRS
        self.netD_drs = netD_drs
        self.dataloader_drs = dataloader_drs
        self.optD_drs = optD_drs
        self.netD_drs_ckpt_file = netD_drs_ckpt_file

        self.topk = topk
        self.gold = gold
        self.gold_step = gold_step

        if self.gold:
            assert self.gold_step is not None

        if self.netD_drs is not None:
            assert self.dataloader_drs is not None and self.optD_drs is not None
            self.train_drs = True
        else:
            self.train_drs = False
        
        # Input values checks
        ints_to_check = {
            'num_steps': num_steps,
            'n_dis': n_dis,
            'print_steps': print_steps,
            'vis_steps': vis_steps,
            'log_steps': log_steps,
            'save_steps': save_steps,
            'flush_secs': flush_secs
        }
        for name, var in ints_to_check.items():
            if var < 1:
                raise ValueError('{} must be at least 1 but got {}.'.format(
                    name, var))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Training helper objects
        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=self.num_steps,
                                    dataset_size=len(self.dataloader),
                                    flush_secs=flush_secs,
                                    device=self.device)

        self.scheduler = DRS_LRScheduler(
            lr_decay=self.lr_decay,
            optimizers=[o for o in [self.optD, self.optD_drs, self.optG] if o is not None],
            num_steps=self.num_steps)

        self.netG_ckpt_dir = os.path.join(self.log_dir, 'checkpoints', 'netG')
        self.netD_ckpt_dir = os.path.join(self.log_dir, 'checkpoints', 'netD')
        self.netD_drs_ckpt_dir = os.path.join(self.log_dir, 'checkpoints', 'netD_drs') if self.train_drs else None

        # Device for hosting model and data
        if not self.device:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else "cpu")

        # Ensure model and data are in the same device
        for net in [self.netD, self.netG, self.netD_drs]:
            if net is not None and net.device != self.device:
                net.to(self.device)
        
    def _save_logit(self, logits_dict):
        for name, logits in logits_dict.items():
            pickle.dump(logits, open(self.output_path / f'logits_{name}.pkl', 'wb'))

    def _get_logit(self, netD, eval_mode=False):
        data_iter = iter(self.dataloader)
        logit_list = np.zeros(len(self.dataloader.dataset))
        if eval_mode:
            netD.eval()
        with torch.no_grad():
            for data, targets, _, idx in data_iter:
                real_data = data.to(self.device)
                logit = netD(real_data)
                if type(logit) is tuple:
                    logit = logit[0]
                logit_r = logit.view(-1)
                logit_list[idx.cpu().numpy()] = logit_r.detach().cpu().numpy()
        netD.train()
        return logit_list

    def _restore_models_and_step(self):
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """
        global_step_D = global_step_G = 0

        if self.netD_ckpt_file:
            assert os.path.exists(self.netD_ckpt_file)
            print("INFO: Restoring checkpoint for D...")
            global_step_D = self.netD.restore_checkpoint(
                ckpt_file=self.netD_ckpt_file, optimizer=self.optD)

        if self.netG_ckpt_file:
            assert os.path.exists(self.netG_ckpt_file)
            print("INFO: Restoring checkpoint for G...")
            global_step_G = self.netG.restore_checkpoint(
                ckpt_file=self.netG_ckpt_file, optimizer=self.optG)

        if self.train_drs and self.netD_drs_ckpt_file:
            assert os.path.exists(self.netD_drs_ckpt_file)
            print("INFO: Restoring checkpoint for D_drs...")
            global_step_D = self.netD_drs.restore_checkpoint(
                ckpt_file=self.netD_drs_ckpt_file, optimizer=self.optD_drs)

        if global_step_D != global_step_G:
            print(f'WARN: global_step_D {global_step_D} != global_step_G {global_step_G}, use global_step_G')
        global_step = global_step_G  # Restores global step

        return global_step

    def _save_model_checkpoints(self, global_step):
        """
        Saves both discriminator and generator checkpoints.
        """
        self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                  global_step=global_step,
                                  optimizer=self.optG)

        if self.netD is not None:
            self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                    global_step=global_step,
                                    optimizer=self.optD)

        if self.train_drs:
            self.netD_drs.save_checkpoint(directory=self.netD_drs_ckpt_dir,
                                  global_step=global_step,
                                  optimizer=self.optD_drs)



    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        
        if self.gold and global_step >= self.gold_step:
            self.netD.use_gold = True

        print("INFO: Starting training from global step {}...".format(
            global_step))
        logit_save_num = 0

        self.logit_results = defaultdict(dict)

        try:
            start_time = time.time()

            # Mixed precision
            if self.amp:
                print("INFO: Using mixed precision training...")
                scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = None

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            if self.train_drs:
                iter_dataloader_drs = iter(self.dataloader_drs)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                if self.topk:
                    self.netG.decay_topk_rate(global_step, epoch_steps=len(self.dataloader))

                if self.gold and global_step == self.gold_step:
                    self.netD.use_gold = True
                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader)

                    # ------------------------
                    #   Update D Network
                    # -----------------------
                    log_data = self.netD.train_step(
                        real_batch=real_batch,
                        netG=self.netG,
                        optD=self.optD,
                        log_data=log_data,
                        global_step=global_step,
                        device=self.device,
                        scaler=scaler)

                    # train netD2 for DRS
                    if self.train_drs:
                        iter_dataloader_drs, real_batch_drs = self._fetch_data(
                            iter_dataloader=iter_dataloader_drs)
                        log_data = self.netD_drs.train_step(
                            real_batch=real_batch_drs,
                            netG=self.netG,
                            optD=self.optD_drs,
                            log_data=log_data,
                            global_step=global_step,
                            device=self.device,
                            scaler=scaler)

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == (self.n_dis - 1):
                        log_data = self.netG.train_step(
                            real_batch=real_batch,
                            netD=self.netD,
                            optG=self.optG,
                            global_step=global_step,
                            log_data=log_data,
                            device=self.device,
                            scaler=scaler)

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                global_step += 1

                log_data = self.scheduler.step(log_data=log_data,
                                               global_step=global_step)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    topk_rate = self.netG.topk_rate if hasattr(self.netG, 'topk_rate') else 1
                    log_data.add_metric(f'topk_rate', topk_rate, group='topk_rate', precision=6)
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                          self.print_steps)
                    start_time = curr_time

                if global_step % self.vis_steps == 0:
                    if 'gaussian' in self.log_dir:
                        plot_gaussian_samples(netG=self.netG,
                                              global_step=global_step,
                                              log_dir=self.log_dir,
                                              device=self.device)
                    else:
                        self.logger.vis_images(netG=self.netG,
                                               global_step=global_step)
                
                if self.save_logits and global_step % self.logit_save_steps == 0 and global_step >= self.save_logit_after and global_step <= self.stop_save_logit_after:
                    if self.train_drs:
                        netD = self.netD_drs
                        netD_name = 'netD_drs'
                    else:
                        netD = self.netD
                        netD_name = 'netD'
                    mode = 'eval' if self.save_eval_logits else 'train'
                    print(f"INFO: logit saving {mode} netD: {netD_name}...")
                    logit_list = self._get_logit(netD=netD, eval_mode=mode=='eval')
                    self.logit_results[f'{netD_name}_{mode}'][global_step] = logit_list

                    logit_save_num += 1

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)
                    if self.save_logits and global_step >= self.save_logit_after:
                        self._save_logit(self.logit_results)

            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)
            if self.save_logits and global_step >= self.save_logit_after:
                self._save_logit(self.logit_results)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)
            if self.save_logits and global_step >= self.save_logit_after:
                self._save_logit(self.logit_results)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")