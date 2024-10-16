#**********************************************************************************
# Copyright (c) 2023 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt-svt/public/GDI-NN
#
# Notes:
# - This code was adpated from the original implementation by Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R. C., & Zavala, V. M. (2023). Capturing molecular interactions in graph neural networks: a case study in multi-component phase equilibrium. Digital Discovery, 2(1), 138-151.
# - The original implementation can be found here: https://github.com/zavalab/ML/tree/master/SolvGNN
#
#*********************************************************************************


from __future__ import absolute_import

# external imports
import sys, random, pickle, csv, time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
import matplotlib.pyplot as plt
import wandb
import argparse

# internal imports
from util.generate_dataset_for_training import solvent_dataset_binary, collate_solvent_binary
from model.model_GNN import solvgnn_binary, solvgnn_xMLP_binary
from model.model_MCM import MCM_multiMLP
from util import data_splitting

class AccumulationMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def train(cv_index, epoch, train_loader, empty_solvsys, model, loss_fn1, loss_fn2, optimizer, pinn_lambda=0, batch_adding=False, wandb_logs=False):
    stage = "train"
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    loss_pred_accum = AccumulationMeter()
    loss1_accum = AccumulationMeter()
    loss2_accum = AccumulationMeter()
    loss_gd_grad_accum = AccumulationMeter()

    # Set model to training mode
    model.train()
    for i, solvdata in enumerate(train_loader):
        end = time.time()
        labgam1 = solvdata['gamma1'].float().cuda() # ln y1
        labgam2 = solvdata['gamma2'].float().cuda() # ln y2
        x1 = solvdata['solv1_x'].float().cuda() # x1
        x2 = 1 - x1 # x2

        # Model predictions and gradients
        y, y1_x1, y2_x1 = None, None, None
        with torch.backends.cudnn.flags(enabled=False):
            y, y1_x1, y2_x1  = model(solvdata, empty_solvsys, gamma_grad=True)   
            
            # Batch adding: Pseudo data is added to data samples in each batch -> only considered for Gibbs-Duhem loss
            if batch_adding:
                # randomly sample x1-x2-pairs from uniform distribution
                add_x1 = torch.distributions.uniform.Uniform(0,1).sample([solvdata["solv1_x"].shape[0],]).cuda()
                add_x2 = 1 - add_x1
                # replace data x1 with pseudo x1
                solvdata["solv1_x"] = add_x1
                # reevaluate model and get gradients
                _, add_y1_x1, add_y2_x1 = model(solvdata, empty_solvsys, gamma_grad=True)
                # Add pseudo data and additional gradients for loss calculation
                x1, x2 = torch.cat([x1, add_x1]), torch.cat([x2, add_x2])
                y1_x1, y2_x1 = torch.cat([y1_x1, add_y1_x1]), torch.cat([y2_x1, add_y2_x1])

        # Gibbs-Duhem loss
        gd_grad = x1 * y1_x1 + x2 * y2_x1
        loss_gd_grad = (gd_grad).pow(2).mean()
        loss_gd_grad_active = loss_gd_grad
        
        # Prediction loss
        loss1 = loss_fn1(y[:,0],labgam1) # loss ln y1
        loss2 = loss_fn2(y[:,1],labgam2) # loss ln y2
        loss_pred = 0.5*loss1+0.5*loss2

        # Overall loss
        loss = loss_pred + pinn_lambda * loss_gd_grad_active
        
        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        loss_accum.update(loss.item(),labgam1.size(0))
        loss_pred_accum.update(loss_pred.item(),labgam1.size(0))
        loss1_accum.update(loss1.item(), labgam1.size(0))
        loss2_accum.update(loss2.item(), labgam2.size(0))
        loss_gd_grad_accum.update(loss_gd_grad.item(), labgam2.size(0))
        batch_time.update(time.time() - end)

        if i % 500 == 0:
            print('Epoch [{}][{}/{}]'
                'Time {:.3f} ({:.3f})\t'
                'Loss {:.3f} ({:.3f})\t'
                'Loss-Pred {:.3f} ({:.3f})\t'
                'Loss1 {:.3f} ({:.3f})\t'
                'Loss2 {:.3f} ({:.3f})\t'
                'Loss-GD {:.6f} ({:.6f})\t'.format(
                epoch, i, len(train_loader), 
                batch_time.value, batch_time.avg,
                loss_accum.value, loss_accum.avg,
                loss_pred_accum.value, loss_pred_accum.avg,
                loss1_accum.value, loss1_accum.avg,
                loss2_accum.value, loss2_accum.avg,
                loss_gd_grad_accum.value, loss_gd_grad_accum.avg,))

    # Logging
    print("[Stage {}]: Epoch {} finished with loss={:.3f} lossPred={:.3f} loss1={:.3f} loss2={:.3f} lossGD={:.6f}".format(
    stage, epoch, loss_accum.avg, loss_pred_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_grad_accum.avg))
    if wandb_logs:
        wandb.log({
                f"epoch": epoch,
                f"train_loss_accum_cv{cv_index}": loss_accum.avg,
                f"train_loss_pred_accum_cv{cv_index}": loss_pred_accum.avg,
                f"train_loss1_accum_cv{cv_index}": loss1_accum.avg,
                f"train_loss2_accum_cv{cv_index}": loss2_accum.avg,
                f"train_loss_gd_accum_cv{cv_index}": loss_gd_grad_accum.avg,
            }, step=epoch)
            
    return [loss_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_grad_accum.avg]


def validate(cv_index, epoch, val_loader, empty_solvsys, model, loss_fn1, loss_fn2, pinn_lambda=0, wandb_logs=False):
    stage = 'validate'
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    loss_pred_accum = AccumulationMeter()
    loss1_accum = AccumulationMeter()
    loss2_accum = AccumulationMeter()
    loss_gd_grad_accum = AccumulationMeter()

    # Set model to eval mode
    model.eval()
    # But we need grads for Gibbs-Duhem loss
    with torch.set_grad_enabled(True):
        for i, solvdata in enumerate(val_loader):
            end = time.time()
            labgam1 = solvdata['gamma1'].float().cuda() # ln y1
            labgam2 = solvdata['gamma2'].float().cuda() # ln y2
            x1 = solvdata['solv1_x'].float().cuda() # x1
            x2 = 1 - x1 # x2

            # Model predictions and gradients
            y, y1_x1, y2_x1 = None, None, None
            with torch.backends.cudnn.flags(enabled=False):
                y, y1_x1, y2_x1 = model(solvdata,empty_solvsys, gamma_grad=True)   

            # Gibbs-Duhem loss
            gd_grad = x1 * y1_x1 + x2 * y2_x1
            loss_gd_grad = (gd_grad).pow(2).mean()
            loss_gd_grad_active = loss_gd_grad
            
            # Prediction loss
            loss1 = loss_fn1(y[:,0],labgam1) # loss ln y1
            loss2 = loss_fn2(y[:,1],labgam2) # loss ln y2
            loss_pred = 0.5*loss1+0.5*loss2

            # Overall loss
            loss = loss_pred + pinn_lambda * loss_gd_grad_active

            # Update stats
            loss_accum.update(loss.item(),labgam1.size(0))
            loss_pred_accum.update(loss_pred.item(),labgam1.size(0))
            loss1_accum.update(loss1.item(), labgam1.size(0))
            loss2_accum.update(loss2.item(), labgam2.size(0))
            loss_gd_grad_accum.update(loss_gd_grad.item(), labgam2.size(0))
            batch_time.update(time.time() - end)

            if i % 500 == 0:
                print('Epoch [{}][{}/{}]'
                    'Time {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'
                    'Loss-Pred {:.3f} ({:.3f})\t'
                    'Loss1 {:.3f} ({:.3f})\t'
                    'Loss2 {:.3f} ({:.3f})\t'
                    'Loss-GD {:.6f} ({:.6f})\t'.format(
                    epoch, i, len(val_loader), 
                    batch_time.value, batch_time.avg,
                    loss_accum.value, loss_accum.avg,
                    loss_pred_accum.value, loss_pred_accum.avg,
                    loss1_accum.value, loss1_accum.avg,
                    loss2_accum.value, loss2_accum.avg,
                    loss_gd_grad_accum.value, loss_gd_grad_accum.avg,))

    # Logging
    print("[Stage {}]: Epoch {} finished with loss={:.3f} lossPred={:.3f} loss1={:.3f} loss2={:.3f} lossGD={:.6f}".format(
            stage, epoch, loss_accum.avg, loss_pred_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_grad_accum.avg))
    if wandb_logs:
        wandb.log({
                    f"epoch": epoch,
                    f"val_loss_accum_cv{cv_index}": loss_accum.avg,
                    f"val_loss_pred_accum_cv{cv_index}": loss_pred_accum.avg,
                    f"val_loss1_accum_cv{cv_index}": loss1_accum.avg,
                    f"val_loss2_accum_cv{cv_index}": loss2_accum.avg,
                    f"val_loss_gd_accum_cv{cv_index}": loss_gd_grad_accum.avg,
                }, step=epoch)

    return [loss_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_grad_accum.avg]


def main(hyperparameter):
    all_start = time.time()

    # fix seed
    seed = hyperparameter.seed

    # model parameters
    model_type = hyperparameter.model_type
    mlp_dropout_rate = hyperparameter.mlp_dropout_rate
    mlp_activation = hyperparameter.mlp_activation
    enc_activation = hyperparameter.enc_activation
    mlp_num_hid_layers = hyperparameter.mlp_num_hid_layers
    hidden_dim = hyperparameter.hidden_dim

    # training parameters 
    pinn_lambda = hyperparameter.pinn_lambda
    pinn_start_epoch = hyperparameter.pinn_start_epoch
    batch_size = hyperparameter.batch_size
    batch_adding = hyperparameter.batch_adding
    lr = hyperparameter.lr
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs

    # data parameters
    data = hyperparameter.data
    data_split_mode = hyperparameter.data_split_mode
    n_splits = hyperparameter.num_splits
    comp_range = hyperparameter.comp_range

    save_add = f"_{data}_split-{data_split_mode}_nums{n_splits}-{model_type}_pinn_l{pinn_lambda}_se{pinn_start_epoch}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_batchA{batch_adding}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}"
    if data_split_mode == "comp_extra":
        save_add = f"_{data}_split-{data_split_mode}_comp{comp_range}-{model_type}_pinn_l{pinn_lambda}_se{pinn_start_epoch}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_batchA{batch_adding}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}"

    
    config = hyperparameter
    wandb_logs = config.wandb_logs
    if wandb_logs:
        if "binaryGamma" == config.data:
            project = "binaryGamma"
        else:
            raise ValueError(f"Wandb project not available for dataset {config.data}")
        wandb.init(config=config, project=project, dir=f"./wandb/{config.data}")


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # read dataset file
    dataset_path = './data/output_binary_with_inf_all.csv'
    solvent_list_path = './data/solvent_list.csv'
    dataset = solvent_dataset_binary(
        input_file_path=dataset_path,
        solvent_list_path=solvent_list_path,
        generate_all=True)
    dataset_size = len(dataset)
    
    # print dataset size
    print('dataset size: {}'.format(dataset_size))

    # get data splits and load dataset
    if data_split_mode == "comp_inter":
        train_indices_splits, val_indices_splits = data_splitting.data_split_comp_inter(
            dataset=dataset, 
            n_splits=n_splits, 
            seed=seed
            )
    elif data_split_mode == "system_extra":
        train_indices_splits, val_indices_splits = data_splitting.data_split_system_extra(
            dataset=dataset, 
            n_splits=n_splits, 
            seed=seed
            )
    elif data_split_mode == "comp_extra":
        train_indices_splits, val_indices_splits = data_splitting.data_split_comp_extra(
            dataset=dataset, 
            comp_range=comp_range,
            )
    else:
        raise NotImplementedError(f"Data splitting {data_split_mode} not implemented.")

    cv_index = 0
    index_list_train = []
    index_list_valid = []    
    for train_indices, val_indices in zip(train_indices_splits, val_indices_splits):
        index_list_train.append(train_indices)
        index_list_valid.append(val_indices)

        # Dataloader
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_solvent_binary,
                                                   shuffle=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler,
                                                 collate_fn=collate_solvent_binary,
                                                 shuffle=False,
                                                 drop_last=True)
        empty_solvsys = dataset.generate_solvsys(batch_size).to("cuda")
        
        # initialize model
        if model_type == "SolvGNN":
            model = solvgnn_binary(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_activation=mlp_activation, mpnn_activation=enc_activation).cuda()
        elif model_type == "SolvGNNxMLP": 
            model = solvgnn_xMLP_binary(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers).cuda()
        elif model_type == "MCM_multiMLP":
            identifier = ["solv1", "solv2"]
            solvent_id_max = dataset.dataset[identifier[0]].str.split("_").str[-1].astype(np.int64).max()
            solute_id_max = dataset.dataset[identifier[1]].str.split("_").str[-1].astype(np.int64).max()
            solvent_id_max = max(solvent_id_max, solute_id_max)
            if enc_activation not in ["relu", "Relu", "RELU", "ReLU"]: 
                raise NotImplementedError(f"Change of activation function {enc_activation} not implemented for MCM encoding yet.")
            if mlp_dropout_rate != 0:
                raise NotImplementedError(f"Change of mlp_dropout_rate {mlp_dropout_rate} not implemented for MCM MLP yet.")
            model = MCM_multiMLP(solvent_id_max=solvent_id_max, solute_id_max=solute_id_max, dim_hidden_channels=hidden_dim, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers).cuda()
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented yet.")

        print(model)
        
        loss_fn1 = nn.MSELoss().cuda()
        loss_fn2 = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        if use_lr_scheduler:
            scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
        # Training
        best_loss = 1000000
        train_loss_save = []
        train_loss1_save = []
        train_loss2_save = []
        train_lossGD_save = []
        val_loss_save = []
        val_loss1_save = []
        val_loss2_save = []
        val_lossGD_save = []
    
        for epoch in range(1, epochs+1):

            # Start Gibbs-Duhem loss in pinn_start_epoch
            tmp_pinn_lambda = 0
            if epoch >= pinn_start_epoch:
                tmp_pinn_lambda = pinn_lambda
            # Train epoch
            train_loss = train(
                cv_index=cv_index, 
                epoch=epoch,
                train_loader=train_loader,
                empty_solvsys=empty_solvsys,
                model=model,
                loss_fn1=loss_fn1,
                loss_fn2=loss_fn2,
                optimizer=optimizer, 
                pinn_lambda=tmp_pinn_lambda, 
                batch_adding=batch_adding, 
                wandb_logs=wandb_logs,
                )
            train_loss_save.append(train_loss[0])
            train_loss1_save.append(train_loss[1])
            train_loss2_save.append(train_loss[2])
            train_lossGD_save.append(train_loss[3])

            # Val epoch
            val_loss = validate(
                cv_index=cv_index, 
                epoch=epoch,
                val_loader=val_loader,
                empty_solvsys=empty_solvsys,
                model=model,
                loss_fn1=loss_fn1,
                loss_fn2=loss_fn2, 
                pinn_lambda=pinn_lambda, 
                wandb_logs=wandb_logs
                )
            val_loss_save.append(val_loss[0])
            val_loss1_save.append(val_loss[1])
            val_loss2_save.append(val_loss[2])
            val_lossGD_save.append(val_loss[3])

            # LR scheduler
            if use_lr_scheduler:
                scheduler.step(train_loss[0])
            
            is_best = val_loss[0] < best_loss
            best_loss = min(val_loss[0], best_loss)
                               
            
        # Save Final model and stats 
        torch.save({
                'epoch': epoch,
                'model_arch': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, './results/final_model_cv{}{}.pth'.format(cv_index, save_add))
        
        np.save('./results/train_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(train_loss_save))
        np.save('./results/train_loss1_cv{}{}.npy'.format(cv_index, save_add),np.array(train_loss1_save))
        np.save('./results/train_loss2_cv{}{}.npy'.format(cv_index, save_add),np.array(train_loss2_save))
        np.save('./results/train_lossGD_cv{}{}.npy'.format(cv_index, save_add),np.array(train_lossGD_save))
        np.save('./results/val_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(val_loss_save))
        np.save('./results/val_loss1_cv{}{}.npy'.format(cv_index, save_add),np.array(val_loss1_save))
        np.save('./results/val_loss2_cv{}{}.npy'.format(cv_index, save_add),np.array(val_loss2_save))
        np.save('./results/val_lossGD_cv{}{}.npy'.format(cv_index, save_add),np.array(val_lossGD_save))
    
        cv_index += 1
    
    np.save(f'./results/train_ind_list{save_add}.npy',index_list_train)
    np.save(f'./results/valid_ind_list{save_add}.npy',index_list_valid)
    
    # Plotting for all cv runs
    ## Training and validation loss
    train_mse = []
    valid_mse = []    
    plt.figure(figsize=(16,8))
    plt_rows = 2 if n_splits > 4 else 1
    for cv_index in range(n_splits):
        train_losses = np.load('./results/train_loss_cv{}{}.npy'.format(cv_index, save_add))
        valid_losses = np.load('./results/val_loss_cv{}{}.npy'.format(cv_index, save_add))
        plt.subplot(plt_rows,int(n_splits/2+0.5)+1,cv_index+1)
        plt.plot(train_losses,label="train loss cv{}".format(cv_index))
        plt.plot(valid_losses,label="valid loss cv{}".format(cv_index))
        plt.xlabel("epoch (training iteration)")
        plt.ylabel("loss")
        plt.legend(loc="best")
        train_mse.append(train_losses[-1])
        valid_mse.append(valid_losses[-1])
    train_mse = np.sqrt(np.array(train_mse))
    valid_mse = np.sqrt(np.array(valid_mse))
    rmse_str = (r'Train RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'Val RMSE = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_mse), np.std(train_mse), np.mean(valid_mse), np.std(valid_mse)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, rmse_str, fontsize=12)
    plt.axis('off')
    plt.savefig(f'./results/cvloss{save_add}_test.png',dpi=300)       
    ## Gibbs-Duhem loss
    train_mse = []
    valid_mse = []   
    plt.figure(figsize=(16,8))
    for cv_index in range(n_splits):
        train_losses = np.load('./results/train_lossGD_cv{}{}.npy'.format(cv_index, save_add))
        valid_losses = np.load('./results/val_lossGD_cv{}{}.npy'.format(cv_index, save_add))
        plt.subplot(plt_rows,int(n_splits/2+0.5)+1,cv_index+1)
        plt.plot(train_losses,label="train loss Gibbs-Duhem cv{}".format(cv_index))
        plt.plot(valid_losses,label="valid loss Gibbs-Duhem cv{}".format(cv_index))
        plt.xlabel("epoch (training iteration)")
        plt.ylabel("Gibbs-Duhem loss")
        plt.legend(loc="best")
        train_mse.append(train_losses[-1])
        valid_mse.append(valid_losses[-1])
    train_mse = np.sqrt(np.array(train_mse))
    valid_mse = np.sqrt(np.array(valid_mse))
    rmse_str = (r'Train Gibbs-Duhem RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'Val Gibbs-Duhem RMSE = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_mse), np.std(train_mse), np.mean(valid_mse), np.std(valid_mse)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, rmse_str, fontsize=12)
    plt.axis('off')
    plt.savefig(f'./results/cvloss_gd{save_add}_test.png',dpi=300)     
    
    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--model_type', default="SolvGNN", type=str) # options: SolvGNN, SolvGNNxMLP, MCM_multiMLP
    parser.add_argument('--batch_size', default=100, type=int) 
    parser.add_argument('--mlp_dropout_rate', default=0.0, type=float)
    parser.add_argument('--mlp_activation', default="softplus", type=str) # 
    parser.add_argument('--enc_activation', default="relu", type=str) # 
    parser.add_argument('--mlp_num_hid_layers', default=2, type=int) # 
    parser.add_argument('--pinn_lambda', default=1.0, type=float) # lambda = factor Gibbs-Duhem loss for overall loss
    parser.add_argument('--pinn_start_epoch', default=0, type=int) # at which epoch pinn loss is considered
    parser.add_argument('--hidden_dim', default=256, type=int) 
    # Training
    parser.add_argument('--batch_adding', default="True", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--use_lr_scheduler', default="True", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    # Data, split, and logs
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--data', default="binaryGamma", type=str)
    parser.add_argument('--data_split_mode', default="comp_inter", type=str)
    parser.add_argument('--num_splits', default=5, type=int)
    parser.add_argument('--comp_range', default=0.0, type=float)
    parser.add_argument('--wandb_logs', default="False", type=str)

    hyperparameter = parser.parse_args()

    # Workaround to set booleans via bash
    if hyperparameter.batch_adding == "False": hyperparameter.batch_adding = False
    if hyperparameter.batch_adding == "True": hyperparameter.batch_adding = True
    if hyperparameter.use_lr_scheduler == "False": hyperparameter.use_lr_scheduler = False
    if hyperparameter.use_lr_scheduler == "True": hyperparameter.use_lr_scheduler = True
    if hyperparameter.wandb_logs == "False": hyperparameter.wandb_logs = False
    if hyperparameter.wandb_logs == "True": hyperparameter.wandb_logs = True
    # Exclude x2 with same compositions as x1 to be excluded
    if hyperparameter.comp_range == 0.0: hyperparameter.comp_range = [0.0, 1.0]
    if hyperparameter.comp_range == 0.1: hyperparameter.comp_range = [0.1, 0.9]
    if hyperparameter.comp_range == 0.3: hyperparameter.comp_range = [0.3, 0.7]
    if hyperparameter.comp_range == 0.5: hyperparameter.comp_range = [0.5, 0.5]

    print(hyperparameter)

    main(hyperparameter=hyperparameter)
