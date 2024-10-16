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
import wandb
import argparse    
import json
import pickle

# internal imports
from model.model_GNN import solvgnn_binary, solvgnn_xMLP_binary
from model.model_MCM import MCM_multiMLP
from util import data_splitting
from util.generate_dataset_for_training import solvent_dataset_binary, collate_solvent_binary

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

def validate(cv_index, val_loader,empty_solvsys,model,loss_fn1,loss_fn2, pinn_lambda=0, gc_weighting=None, wandb_logs=False):
    stage = 'validate'
    loss_accum = AccumulationMeter()
    loss_pred_accum = AccumulationMeter()
    loss1_accum = AccumulationMeter()
    loss2_accum = AccumulationMeter()
    loss_gd_accum = AccumulationMeter()
    val_pred = torch.tensor([]).cpu()
    val_true = torch.tensor([]).cpu()
    gd = torch.tensor([]).cpu()
    model.eval()
    x1_list = torch.tensor([])
    with torch.set_grad_enabled(True):
        for i, solvdata in enumerate(val_loader):
            labgam1 = solvdata['gamma1'].float().cuda()
            labgam2 = solvdata['gamma2'].float().cuda()
            output = None
            with torch.backends.cudnn.flags(enabled=False):
                output, y1_x1, y2_x1 = model(solvdata,empty_solvsys, gamma_grad=True)   
            x1 = solvdata['solv1_x'].float().cuda()
            x2 = 1 - x1
            x1_list = torch.concatenate([x1_list, x1.detach().cpu()])
            gd_grad = x1 * y1_x1 + x2 * y2_x1
            loss_gd_grad = (gd_grad).pow(2).mean()
            loss1 = loss_fn1(output[:,0],labgam1)
            loss2 = loss_fn2(output[:,1],labgam2)
            loss = 0.5*loss1+0.5*loss2+pinn_lambda*loss_gd_grad
            loss_pred = 0.5*loss1+0.5*loss2
            loss_accum.update(loss.item(),labgam1.size(0))
            loss_pred_accum.update(loss_pred.item(),labgam1.size(0))
            loss1_accum.update(loss1.item(), labgam1.size(0))
            loss2_accum.update(loss2.item(), labgam2.size(0))
            loss_gd_accum.update(loss_gd_grad.item(), labgam2.size(0))
            val_pred = torch.concatenate([val_pred, output.detach().cpu()])
            val_true = torch.concatenate([val_true, torch.stack([labgam1, labgam2], axis=1).detach().cpu()])
            gd = torch.concatenate([gd, gd_grad.detach().cpu()])

    #breakpoint()
    print("[Stage {}]: loss={:.3f} loss1={:.3f} loss2={:.3f} lossGD={:.6f} ".format(
            stage, loss_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_accum.avg))

    return val_pred, val_true, x1_list, gd, [loss_accum.avg, loss_pred_accum.avg, loss1_accum.avg, loss2_accum.avg, loss_gd_accum.avg]


def main(hyperparameter):

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
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs
    lr = hyperparameter.lr

    # data parameters
    data = hyperparameter.data
    data_split_mode = hyperparameter.data_split_mode
    n_splits = hyperparameter.num_splits
    comp_range = hyperparameter.comp_range

    save_add = f"_{data}_split-{data_split_mode}_nums{n_splits}-{model_type}_pinn_l{pinn_lambda}_se{pinn_start_epoch}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_batchA{batch_adding}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}"
    if data_split_mode == "comp_extra":
        save_add = f"_{data}_split-{data_split_mode}_comp{comp_range}-{model_type}_pinn_l{pinn_lambda}_se{pinn_start_epoch}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_batchA{batch_adding}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

    if hyperparameter.external_data == None: 
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
    else:
        # data set for composition test ext
        if hyperparameter.external_data == "all_systems_comp_range_step5e-2":
            # read dataset file
            dataset_path = './data/all_systems_comp_range_step5e-2.csv'
        else:
            raise ValueError(f"External data set - {hyperparameter.external_data} - not available for evaluation ")
        solvent_list_path = './data/solvent_list.csv'
        dataset = solvent_dataset_binary(
            input_file_path=dataset_path,
            solvent_list_path=solvent_list_path,
            generate_all=True)
        dataset_size = len(dataset)
        train_indices_splits = [[]]*n_splits
        val_indices_splits = [[i for i in range(dataset.dataset.shape[0])]]*n_splits
    

    cv_index = 0
    index_list_train = []
    index_list_valid = []
    val_pred_cvs = []
    val_true_cvs = []
    val_gd_cvs = []
    val_x1_list_cvs = []
    val_loss_cvs = []
    val_loss_pred_cvs = []
    val_loss1_cvs = []
    val_loss2_cvs = []
    val_lossGD_cvs = []

    df_dataset = dataset.dataset.copy()
    df_dataset["pred_lngam1"] = None
    df_dataset["pred_lngam2"] = None
    df_dataset["GD_deviation"] = None
    df_dataset["GDw_deviation"] = None

    for train_indices, val_indices in zip(train_indices_splits, val_indices_splits):

        print(f"Evaluate cv {cv_index}")

        index_list_train.append(train_indices)
        index_list_valid.append(val_indices)

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)

        # Dataloader
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                 #sampler=valid_sampler,
                                                 collate_fn=collate_solvent_binary,
                                                 shuffle=False,
                                                 drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                                 #sampler=valid_sampler,
                                                 collate_fn=collate_solvent_binary,
                                                 shuffle=False,
                                                 drop_last=True)
        print(len(val_loader))
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


        model.load_state_dict(torch.load('./results/final_model_cv{}{}.pth'.format(cv_index, save_add))["model_state_dict"])


        loss_fn1 = nn.MSELoss().cuda()
        loss_fn2 = nn.MSELoss().cuda()        
        val_pred, val_true, x1_list, val_gd, val_loss = validate(cv_index, val_loader,empty_solvsys,model,loss_fn1,loss_fn2, pinn_lambda)
        val_pred_cvs.append(val_pred)
        val_true_cvs.append(val_true)
        val_gd_cvs.append(val_gd)
        val_x1_list_cvs.append(x1_list)
        val_loss_cvs.append(val_loss[0])
        val_loss_pred_cvs.append(val_loss[1])
        val_loss1_cvs.append(val_loss[2])
        val_loss2_cvs.append(val_loss[3])
        val_lossGD_cvs.append(val_loss[4])

        if hyperparameter.external_data == None:
            df_dataset.loc[val_indices, "pred_lngam1"] = val_pred[:,0].numpy()
            df_dataset.loc[val_indices, "pred_lngam2"] = val_pred[:,1].numpy()
            df_dataset.loc[val_indices, "GD_deviation"] = val_gd

        cv_index += 1
        break

    if hyperparameter.external_data == None:
        df_dataset.to_csv(f"./analysis/predictions/preds{save_add}.csv.gzip", compression='gzip')
    gd_for_x1_dict = {}
    for l_idx, x1s in enumerate(val_x1_list_cvs):
        unique_x1s = torch.unique(x1s).tolist()
        for x1_value in unique_x1s:
            x1_mask = torch.where(val_x1_list_cvs[l_idx] == x1_value)
            mse_gd_for_x1 = val_gd_cvs[l_idx][x1_mask].pow(2).mean().unsqueeze(-1)
            mae_gd_for_x1 = val_gd_cvs[l_idx][x1_mask].abs().mean().unsqueeze(-1)
            if x1_value in gd_for_x1_dict.keys():
                gd_for_x1_dict[x1_value]["mse"] = np.concatenate([gd_for_x1_dict[x1_value]["mse"], mse_gd_for_x1.numpy()])
                gd_for_x1_dict[x1_value]["mae"] = np.concatenate([gd_for_x1_dict[x1_value]["mae"], mae_gd_for_x1.numpy()])
            else:
                gd_for_x1_dict[x1_value] = {}
                gd_for_x1_dict[x1_value]["mse"] = mse_gd_for_x1.numpy()
                gd_for_x1_dict[x1_value]["mae"] = mae_gd_for_x1.numpy()

    gd_for_x1_stats = {}
    for x1, metrics in gd_for_x1_dict.items():
        for m, x1_stats in metrics.items():
            gd_for_x1_stats[f"{m}_GD_for_solv1x{round(x1,2)}_mean"] = np.mean(x1_stats, dtype=np.float64)
            gd_for_x1_stats[f"{m}_GD_for_solv1x{round(x1,2)}_std"] = np.std(x1_stats,ddof=1, dtype=np.float64)


    stats = {
        "mse_loss_mean": np.mean(np.array(val_loss_cvs)),
        "mse_loss_std": np.std(np.array(val_loss_cvs),ddof=1),
        "mse_lngam_mean": np.mean(np.array(val_loss_pred_cvs)),
        "mse_lngam_std": np.std(np.array(val_loss_pred_cvs),ddof=1),
        "rmse_lngam_mean": np.mean(np.array(val_loss_pred_cvs)**(1/2)),
        "rmse_lngam_std": np.std(np.array(val_loss_pred_cvs)**(1/2),ddof=1),
        "mse_lngam1_mean": np.mean(np.array(val_loss1_cvs)),
        "mse_lngam1_std": np.std(np.array(val_loss1_cvs),ddof=1),
        "mse_lngam2_mean": np.mean(np.array(val_loss2_cvs)),
        "mse_lngam2_std": np.std(np.array(val_loss2_cvs),ddof=1),
        "mse_gd_mean": np.mean(np.array(val_lossGD_cvs)),
        "mse_gd_std": np.std(np.array(val_lossGD_cvs),ddof=1),
        "rmse_gd_mean": np.mean(np.array(val_lossGD_cvs)**(1/2)),
        "rmse_gd_std": np.std(np.array(val_lossGD_cvs)**(1/2),ddof=1),
    }
    
    print(stats)
    print(gd_for_x1_stats)

    with open(f"./analysis/predictions/stats{save_add}_{dataset_path.split('/')[-1].split('.')[0]}.json", 'w+') as f:
        json.dump(stats, f)
    with open(f"./analysis/predictions/stats_GD{save_add}_{dataset_path.split('/')[-1].split('.')[0]}.json", 'w+') as f:
        json.dump(gd_for_x1_stats, f)
    with open(f"./analysis/predictions/stats_GD{save_add}_{dataset_path.split('/')[-1].split('.')[0]}.pkl", 'wb') as f:
        pickle.dump(gd_for_x1_dict, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # GNN architecture
    parser.add_argument('--model_type', default="SolvGNN", type=str)
    parser.add_argument('--pinn_lambda', default=1.0, type=float)
    parser.add_argument('--pinn_start_epoch', default=0, type=int) # at which epoch pinn loss is considered
    parser.add_argument('--batch_size', default=2000, type=int) 
    parser.add_argument('--mlp_dropout_rate', default=0.0, type=float)
    parser.add_argument('--mlp_activation', default="softplus", type=str) # 
    parser.add_argument('--enc_activation', default="relu", type=str) # 
    parser.add_argument('--mlp_num_hid_layers', default=2, type=int) # 
    parser.add_argument('--hidden_dim', default=256, type=int) 
    parser.add_argument('--batch_adding', default="True", type=str)
    parser.add_argument('--use_lr_scheduler', default="True", type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    # Data, split, and logs
    parser.add_argument('--wandb_logs', default=True, type=bool)
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--data', default="binaryGamma", type=str)
    parser.add_argument('--data_split_mode', default="comp_inter", type=str)
    parser.add_argument('--num_splits', default=5, type=int)
    parser.add_argument('--comp_range', default=[0.3, 0.7], type=list)
    parser.add_argument('--external_data', default="all_systems_comp_range_step5e-2", type=str)
    hyperparameter = parser.parse_args()

    if hyperparameter.batch_adding == "False": hyperparameter.batch_adding = False
    if hyperparameter.batch_adding == "True": hyperparameter.batch_adding = True
    if hyperparameter.use_lr_scheduler == "False": hyperparameter.use_lr_scheduler = False
    if hyperparameter.use_lr_scheduler == "True": hyperparameter.use_lr_scheduler = True
    if hyperparameter.wandb_logs == "False": hyperparameter.wandb_logs = False
    if hyperparameter.wandb_logs == "True": hyperparameter.wandb_logs = True
    if hyperparameter.external_data == "None": hyperparameter.external_data = None

    print(hyperparameter)

    main(hyperparameter=hyperparameter)
