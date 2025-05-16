import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import norm

from copy import deepcopy
import itertools
import random
from collections import OrderedDict

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import multiprocessing


from ses_network_3_0 import SESNetwork
#from extraction_works.ses_network_2_0_copy import SESNetwork
from utils import make_input, LatentSpace, get_sample_from_num_swaps, get_cos_sim_np, get_cos_sim_torch, test_network, get_ordered_weights


def save_results(results_list, filename):


    results = {}
    results["ordered_indices_ctx"] = []
    results["ordered_indices_mtl"] = []
    results["ordered_indices_mtl_dense"] = []
    results["ordered_indices_mtl_sparse"] = []
    results["selectivity_ctx"] = []
    results["selectivity_mtl"] = []
    results["selectivity_mtl_dense"] = []
    results["selectivity_mtl_sparse"] = []
    results["ctx_ctx"] = []
    results["ctx_mtl"] =[]
    results["accuracy_mtl_sparse"] = []


    for ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy_A, accuracy_B, ctx_ctx,  ctx_mtl in results_list:


        results["ordered_indices_ctx"].append(ordered_indices_ctx)
        results["ordered_indices_mtl"].append(ordered_indices_mtl)
        results["ordered_indices_mtl_dense"].append(ordered_indices_mtl_dense)
        results["ordered_indices_mtl_sparse"].append(ordered_indices_mtl_sparse)
        results["selectivity_ctx"].append(selectivity_ctx)
        results["selectivity_mtl"].append(selectivity_mtl)
        results["selectivity_mtl_dense"].append(selectivity_mtl_dense)
        results["selectivity_mtl_sparse"].append(selectivity_mtl_sparse)
        results["ctx_ctx"].append(ctx_ctx)
        results["ctx_mtl"].append(ctx_mtl)
        results["accuracy_mtl_sparse"].append((accuracy_A, accuracy_B))

    with open('Data/{}'.format(filename), 'wb') as handle:
        pickle.dump(dict(results), handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_network(net, input_params, print_rate=1):
  input, input_episodes, input_latents = make_input(**input_params)
  with torch.no_grad():
    for day in range(input_params["num_days"]):
      if day%print_rate == 0:
        print(day)
      net(input[day], debug=False)
      net.sleep()
  return input, input_episodes, input_latents, net


def blokced_interleaved(network_parameters, recording_parameters, input_params, latent_specs, training='blocked', seed=42):

    
    if training=='interleaved':


        input_params["num_days"] = 2000
        input_params["day_length"] = 40
        input_params["mean_duration"] = 5
        input_params["num_swaps"] = 8
        input_params["latent_space"] = LatentSpace(**latent_specs)
        network = SESNetwork(network_parameters, recording_parameters)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

    if training == 'blocked':

        original_prob_list = deepcopy(latent_specs["prob_list"])

        input_params["num_days"] = 10
        input_params["day_length"] = 200
        input_params["mean_duration"] = 1

        latent_specs["prob_list"] = [0.2 if i==0 else 0 for i in range(5) for j in range(5)]
        input_params["latent_space"] = LatentSpace(**latent_specs)

        network = SESNetwork(network_parameters, recording_parameters)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

        for k in range(1, 5):
            latent_specs["prob_list"] = [0.2 if i==k else 0 for i in range(5) for j in range(5)]
            input_params["latent_space"] = LatentSpace(**latent_specs)
            input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

        for k in range(5):
            latent_specs["prob_list"] = [0.2 if j==k else 0 for i in range(5) for j in range(5)]
            input_params["latent_space"] = LatentSpace(**latent_specs)
            input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)


        input_params["num_days"] = 1900
        input_params["day_length"] = 40
        input_params["mean_duration"] = 5
        latent_specs["prob_list"] = original_prob_list
        input_params["latent_space"] = LatentSpace(**latent_specs)
        input, input_episodes, input_latents, network = test_network(network, input_params, print_rate=50)

    ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy_A, accuracy_B = get_selectivity_accuracy(network, input_latents, input_episodes, input_params, latent_specs)
 
    return ordered_indices_ctx, ordered_indices_mtl, ordered_indices_mtl_dense, ordered_indices_mtl_sparse, selectivity_ctx, selectivity_mtl, selectivity_mtl_dense, selectivity_mtl_sparse, accuracy_A, accuracy_B, network.ctx_ctx,  network.ctx_mtl




network_parameters = {}

network_parameters["hebbian_filter"] = False

network_parameters["duration_phase_A"] = 1000
network_parameters["duration_phase_B"] = 1500

network_parameters["sleep_duration_A"] = 10
network_parameters["sleep_duration_B"] = 10
network_parameters["reset_dayly"] = True

network_parameters["regions"] = ["sen", "mtl_sparse", "mtl_dense", "mtl", "ctx"]

network_parameters["mtl_pattern_complete_iterations"] = 10
network_parameters["mtl_dense_pattern_complete_iterations"] = 5
network_parameters["mtl_sparse_pattern_complete_iterations"] = 10
network_parameters["ctx_pattern_complete_iterations"] = 10
network_parameters["mtl_generate_pattern_complete_iterations"] = 10

network_parameters["max_semantic_charge_replay"] = 1
network_parameters["max_semantic_charge_input"] = 2

network_parameters["sen_num_subregions"] = 1
network_parameters["sen_size_subregions"] = torch.tensor([100])
network_parameters["sen_sparsity"] = torch.tensor([0.2])
network_parameters["sen_sparsity_sleep"] = torch.tensor([0.2])

network_parameters["ctx_num_subregions"] = 2
network_parameters["ctx_size_subregions"] =  torch.tensor([100, 250])
network_parameters["ctx_sparsity"] = torch.tensor([0.2, 1/25])
network_parameters["ctx_sparsity_sleep"] = torch.tensor([0.1, 1/25])

network_parameters["mtl_num_subregions"] = 2
network_parameters["mtl_size_subregions"] =  torch.tensor([100, 100])
network_parameters["mtl_sparsity"] = torch.tensor([0.2, 0.1])
network_parameters["mtl_sparsity_sleep"] = torch.tensor([0.1, 0.05])

network_parameters["mtl_dense_num_subregions"] = 1
network_parameters["mtl_dense_size_subregions"] = torch.tensor([100])
network_parameters["mtl_dense_sparsity"] = torch.tensor([0.2])
network_parameters["mtl_dense_sparsity_sleep"] = torch.tensor([0.1])

network_parameters["mtl_sparse_num_subregions"] = 1
network_parameters["mtl_sparse_size_subregions"] = torch.tensor([100])
network_parameters["mtl_sparse_sparsity"] = torch.tensor([0.1])
network_parameters["mtl_sparse_sparsity_sleep"] = torch.tensor([0.05])

network_parameters["mtl_dense_sen_projection"] = True
network_parameters["mtl_dense_sen_size"] = 30
network_parameters["max_post_mtl_dense_sen"] = 1
network_parameters["max_pre_mtl_dense_sen"] = np.inf


network_parameters["ctx_mtl_quick_lmbda"] = 1e-2
network_parameters["max_pre_ctx_mtl_quick"] = np.inf
network_parameters["max_post_ctx_mtl_quick"] = 1

network_parameters["ctx_mtl_sparsity"] = 0.5
network_parameters["ctx_mtl_mean"] = 0.03
network_parameters["ctx_mtl_std"] = 0.005
network_parameters["ctx_mtl_lmbda"] = 5e-4
network_parameters["ctx_mtl_size"] = 2
network_parameters["max_pre_ctx_mtl"] = np.inf
network_parameters["max_post_ctx_mtl"] = 1

network_parameters["ctx_mtl_sparse_mean"] = 0.07
network_parameters["ctx_mtl_sparse_std"] = 0.001
network_parameters["ctx_mtl_sparse_lmbda"] = 5e-4
network_parameters["max_pre_ctx_mtl_sparse"] = np.inf
network_parameters["max_post_ctx_mtl_sparse"] = 1

network_parameters["ctx_mtl_dense_sparsity"] = 0.5
network_parameters["ctx_mtl_dense_g"] = 0.01

network_parameters["mtl_mtl_lmbda"] = 5e-3
network_parameters["max_pre_mtl_mtl"] = np.inf
network_parameters["max_post_mtl_mtl"] = np.inf

network_parameters["mtl_dense_mtl_dense_lmbda"] = 5e-3
network_parameters["max_pre_mtl_dense_mtl_dense"] = np.inf
network_parameters["max_post_mtl_dense_mtl_dense"] = np.inf

network_parameters["mtl_sparse_mtl_sparse_lmbda"] = 5e-3
network_parameters["max_pre_mtl_sparse_mtl_sparse"] = np.inf
network_parameters["max_post_mtl_sparse_mtl_sparse"] = np.inf

network_parameters["ctx_ctx_sparsity"] = 0.05
network_parameters["ctx_ctx_g"] = 1e-4
network_parameters["ctx_ctx_lmbda"] = 5e-4
network_parameters["max_pre_ctx_ctx"] = 1
network_parameters["max_post_ctx_ctx"] = np.inf

network_parameters["mtl_sparse_ctx_mean"] = 0.03
network_parameters["mtl_sparse_ctx_std"] = 0.001
network_parameters["mtl_sparse_ctx_lmbda"] = 5e-3
network_parameters["max_pre_mtl_sparse_ctx"] = np.inf
network_parameters["max_post_mtl_sparse_ctx"] = 1


recording_parameters = {}
recording_parameters["regions"] = ["sen", "mtl_hat", "mtl_dense", "mtl_sparse", "mtl_sparse_hat", "mtl", "ctx", "ctx_hat"]
recording_parameters["rate_activity"] = 1
#recording_parameters["connections"] = ["mtl_mtl", "ctx_mtl", "ctx_ctx", "ctx_mtl_quick", "mtl_sparse_ctx"]
recording_parameters["connections"] = []
recording_parameters["rate_connectivity"] = np.inf


input_params = {}
input_params["num_days"] = 1
input_params["day_length"] = 40
input_params["mean_duration"] = 5
input_params["fixed_duration"] = True
input_params["num_swaps"] = 8


latent_specs = {}
latent_specs["num"] = 2
latent_specs["total_sizes"] = [50, 50]
latent_specs["act_sizes"] = [10, 10]
latent_specs["dims"] = [5, 5]
latent_specs["prob_list"] = [1/25 for i in range(5) for j in range(5)]


num_cpu = 20
trainings = ["interleaved", "blocked"]
num_seeds = 5
seeds = np.arange(num_seeds)


experiment_params = [(network_parameters, recording_parameters, input_params, latent_specs, training, seed) for training in trainings for seed in seeds]
pool = multiprocessing.Pool(processes=num_cpu)

results_list = pool.starmap(blokced_interleaved, experiment_params)
save_results(results_list, filename='fig_5_blokced_interleaved.pickle')


