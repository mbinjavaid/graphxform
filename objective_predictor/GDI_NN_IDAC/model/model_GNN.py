# -*- coding: utf-8 -*-

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


import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import torch.nn.functional as F


def get_n_params(model):
    n_params = 0
    for item in list(model.parameters()):
        item_param = 1
        for dim in list(item.size()):
            item_param = item_param*dim
        n_params += item_param
    return n_params

def get_activation(activation, get_nn=False):
    if (activation == None) or (activation in ["relu", "ReLU", "RELU"]):
        if get_nn: return nn.ReLU
        return F.relu
    elif activation in ["elu", "ELU"]:
        if get_nn: return nn.ELU
        return F.elu
    elif activation in ["LeakyReLU", "LeakyRELU", "leakyReLU", "leakyrelu", "leakyRELU", "leaky_relu", "Leaky_ReLU", "Leaky_RELU"]:
        if get_nn: return nn.LeakyReLU
        return F.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn: return nn.Sigmoid
        return F.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn: return nn.Softplus
        return F.softplus


class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6, activation="relu"):
        super(MPNNconv, self).__init__()

        self.mpnn_activation = get_activation(activation)
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            get_activation(activation, get_nn=True)()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            get_activation(activation, get_nn=True)(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats    

class solvgnn_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim+1,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                #solv2x = 1 - solv1x
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()
                
                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
        
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                output = self.mlp_activation(self.classify1(hg))
                output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                        
                output = torch.cat(
                    (output[0:len(output)//2,:],
                    output[len(output)//2:,:]),axis=1)      
                    
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]   
                    return output, y1_x1, y2_x1                          
            return output         


class solvgnn_xMLP_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_xMLP_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim+1, hidden_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg, torch.cat((solv1x, 1-solv1x))[:, None]), axis=1)
                
                output = self.mlp_dropout(hg)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                        
                output = torch.cat(
                    (output[0:len(output)//2,:],
                    output[len(output)//2:,:]),axis=1)    
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    


class solvgnn_onexMLP_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_dropout(hg)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    

class solvgnn_onexMLP_share1layer_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_share1layer_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
            self.classify2_2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3_1 = nn.Linear(hidden_dim*2, n_classes)
        self.classify3_2 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_activation(self.classify1(hg))
                output_y1 = self.mlp_activation(self.classify2_1(output))
                output_y1 = self.classify3_1(output_y1)
                output_y2 = self.mlp_activation(self.classify2_2(output))
                output_y2 = self.classify3_2(output_y2)
                output = torch.cat([output_y1, output_y2], dim=1)  
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    

class solvgnn_onexMLP_share2layer_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_share2layer_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3_1 = nn.Linear(hidden_dim*2, n_classes)
        self.classify3_2 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_activation(self.classify1(hg))
                output = self.mlp_activation(self.classify2(output))
                output_y1 = self.classify3_1(output)
                output_y2 = self.classify3_2(output)
                output = torch.cat([output_y1, output_y2], dim=1)  
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    


