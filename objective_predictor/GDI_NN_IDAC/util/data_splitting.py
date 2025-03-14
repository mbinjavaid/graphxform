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
# - The comp-inter code was adpated from the original implementation by Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R. C., & Zavala, V. M. (2023). Capturing molecular interactions in graph neural networks: a case study in multi-component phase equilibrium. Digital Discovery, 2(1), 138-151.
#
#*********************************************************************************


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def data_split_comp_inter(dataset, n_splits=5, seed=2021):
    train_indices_splits = []
    val_indices_splits = []
    
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    tpsa_binary = dataset.dataset['tpsa_binary_avg'].to_numpy()
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind,tpsa_binary):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits

def data_split_system_extra(dataset, n_splits=5, seed=2021):
    train_indices_splits = []
    val_indices_splits= []

    bin_data = dataset.dataset
    bin_data_red_tpsa = bin_data.groupby(["solv1", "solv2"], as_index=False)["tpsa_binary_avg"].mean()
    tpsa_binary = bin_data_red_tpsa['tpsa_binary_avg'].to_numpy()
    num_systems = len(bin_data_red_tpsa)
    all_sys_ind = np.arange(num_systems)
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_sys_ind, val_sys_ind in kf.split(all_sys_ind,tpsa_binary):
        train_sys = bin_data_red_tpsa[bin_data_red_tpsa.index.isin(train_sys_ind)]
        train_bin_df = bin_data.reset_index().merge(train_sys, on=["solv1", "solv2"]).set_index("index")
        train_indices = train_bin_df.index.tolist()
        val_sys = bin_data_red_tpsa[bin_data_red_tpsa.index.isin(val_sys_ind)]
        val_bin_df = bin_data.reset_index().merge(val_sys, on=["solv1", "solv2"]).set_index("index") #bin_data.merge(val_sys, on=["solv1", "solv2"])
        val_indices = val_bin_df.index.to_list()
        
        # sanity check
        train_sys_tuples = list(zip(train_bin_df.solv1, train_bin_df.solv2))
        val_sys_tuples = list(zip(val_bin_df.solv1, val_bin_df.solv2))
        train_val_sys_intersect = set(train_sys_tuples).intersection(val_sys_tuples)
        train_val_indices_intersect = set(train_indices).intersection(val_indices)
        if (bool(train_val_sys_intersect) == True) or (bool(train_val_indices_intersect) == True):
            raise ValueError("Something with the data splitting went wrong. System in validation set is also in training set.")

        train_indices_splits.append(train_indices)
        val_indices_splits.append(val_indices)
    return train_indices_splits, val_indices_splits

def data_split_comp_extra(dataset, comp_range=[0.5]):
    train_indices_list = []
    val_indices_list = []
    excl_range = comp_range
    bin_data = dataset.dataset
    train_bin_df = bin_data[~bin_data["solv1_x"].isin(excl_range)]
    train_indices = train_bin_df.index.tolist()
    val_bin_df = bin_data[bin_data["solv1_x"].isin(excl_range)]
    val_indices = val_bin_df.index.tolist()

    # sanity check
    train_val_indices_intersect = set(train_indices).intersection(val_indices)
    if (bool(train_val_indices_intersect) == True):
        raise ValueError("Something with the data splitting went wrong. System in validation set is also in training set.")

    train_indices_list.append(train_indices)
    val_indices_list.append(val_indices)
    return train_indices_list, val_indices_list
