
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from collections import OrderedDict

import matplotlib.pyplot as plt


class SESNetwork(nn.Module):
    def __init__(self, net_params, rec_params):

      super(SESNetwork, self).__init__()
      self.init_network(net_params)
      self.init_recordings(rec_params)

    def forward(self, input, debug=False):
        if self.reset_dayly:
            self.mtl_mtl_plastic = torch.zeros((self.mtl_size, self.mtl_size))
            self.mtl_sparse_mtl_sparse_plastic = torch.zeros((self.mtl_sparse_size, self.mtl_sparse_size))
            self.mtl_dense_mtl_dense_plastic =  torch.zeros((self.mtl_dense_size, self.mtl_dense_size))

        for timestep in range(input.shape[0]):

            self.sen, _ = self.activation(input[timestep], 'sen')

            self.mtl_dense_hat = F.linear(self.sen, self.mtl_dense_sen)
            self.mtl_dense, _ = self.activation(self.mtl_dense_hat, 'mtl_dense')

            self.mtl_sparse_hat = torch.randn(self.mtl_sparse_size)
            self.mtl_sparse, _ = self.activation(self.mtl_sparse_hat, 'mtl_sparse')

            self.mtl[:self.mtl_dense_size] = self.mtl_dense
            self.mtl[self.mtl_dense_size:] = self.mtl_sparse

            self.ctx_hat = F.linear(self.mtl, self.ctx_mtl)
            self.ctx, _ = self.activation(self.ctx_hat, 'ctx')

            if self.day >= self.duration_phase_A:
               self.ctx = self.pattern_complete('ctx', self.ctx)
               self.mtl_sparse_hat = F.linear(self.ctx, self.mtl_sparse_ctx)
               self.mtl_sparse, _ = self.activation(self.mtl_sparse_hat, 'mtl_sparse')
               self.mtl[self.mtl_dense_size:] = self.mtl_sparse

               self.hebbian('mtl_sparse', 'mtl_sparse')
               self.homeostasis('mtl_sparse', 'mtl_sparse')


            if self.day >= self.duration_phase_B:
               self.ctx_hat = F.linear(self.mtl, self.ctx_mtl)
               self.ctx, _ = self.activation(self.ctx_hat, 'ctx')



            if debug:
              plt.imshow(self.mtl_sparse_mtl_sparse)

            self.hebbian('mtl', 'mtl')
            self.homeostasis('mtl', 'mtl')

            self.hebbian('mtl_dense', 'mtl_dense')
            self.homeostasis('mtl_dense', 'mtl_dense')

            self.hebbian('ctx', 'ctx')
            self.homeostasis('ctx', 'ctx')

            #self.hebbian('ctx', 'ctx', inh=True)
            #self.homeostasis('ctx', 'ctx', inh=True)

            self.record()
            self.time_index += 1
            self.awake_indices.append(self.time_index)
        self.day += 1

    def sleep(self):
      for timestep in range(self.sleep_duration_A):

        if self.day <= self.duration_phase_B:
          mtl_dense_random = torch.randn(self.mtl_dense_size)**2
          self.mtl_dense = self.pattern_complete('mtl_dense', h_0=mtl_dense_random, sleep=True)
          self.mtl[:self.mtl_dense_size] = self.mtl_dense
          self.mtl[self.mtl_dense_size:] = 0
          self.ctx_hat = F.linear(self.mtl, self.ctx_mtl)
          self.ctx, _ = self.activation(self.ctx_hat, 'ctx', subregion_index=0, sleep=True)

          self.hebbian('ctx', 'mtl')
          self.homeostasis('ctx', 'mtl')

          if self.day >= self.duration_phase_A:
                self.ctx = self.pattern_complete('ctx', self.ctx, sleep=True)

        else:
          semantic_charge = torch.randint(low=1, high=self.max_semantic_charge_replay+1, size=(1,))[0]
          self.mtl = self.mtl_generate(semantic_charge)
          self.ctx_hat = F.linear(self.mtl, self.ctx_mtl)
          self.ctx, _ = self.activation(self.ctx_hat, 'ctx', subregion_index=semantic_charge-1, sleep=True)

          self.hebbian('ctx', 'mtl')
          self.homeostasis('ctx', 'mtl')



        self.hebbian('ctx', 'ctx')
        self.homeostasis('ctx', 'ctx')

        self.record()
        self.time_index += 1
        self.sleep_indices_A.append(self.time_index)

      if self.day >= self.duration_phase_A:

        for timestep in range(self.sleep_duration_B):

          ctx_random = torch.randn(self.ctx_size)
          self.ctx = self.pattern_complete('ctx', h_0=ctx_random, subregion_index=0, sleep=True)

          self.mtl_sparse_hat = F.linear(self.ctx, self.mtl_sparse_ctx)
          self.mtl_sparse, _ = self.activation(self.mtl_sparse_hat, 'mtl_sparse', sleep=True)
          self.mtl[self.mtl_dense_size:] = self.mtl_sparse

          self.hebbian('mtl_sparse', 'ctx')
          self.homeostasis('mtl_sparse', 'ctx')

          self.record()
          self.time_index += 1
          self.sleep_indices_B.append(self.time_index)

    def activation(self, x, region, x_conditioned=None, subregion_index=None, sleep=False, sparsity=None):
      x = x + (1e-10 + torch.max(x) - torch.min(x))/100*torch.randn(x.shape)

      if x_conditioned is not None:
         x[x_conditioned==1] = torch.max(x) + 1
      x_prime = torch.zeros(x.shape)
      x_sparsity = getattr(self, region + '_sparsity') if not sleep else getattr(self, region + '_sparsity_sleep')
      x_sparsity = x_sparsity if sparsity is None else sparsity 
      x_subregions = getattr(self, region + '_subregions')

      if sleep:
        subregional_input = [x[subregion].sum() for subregion in x_subregions]
        subregion_index = torch.topk(torch.tensor(subregional_input), 1).indices.int() if subregion_index is None else subregion_index
        subregion = x_subregions[subregion_index]
        x_subregion = torch.zeros_like(subregion).float()
        top_indices = torch.topk(x[subregion], int(len(subregion)*x_sparsity[subregion_index])).indices
        x_subregion[top_indices] = 1
        x_prime[subregion]  = x_subregion

      else:
        for subregion_index, subregion in enumerate(x_subregions):
          x_subregion = torch.zeros_like(subregion).float()
          top_indices = torch.topk(x[subregion], int(len(subregion)*x_sparsity[subregion_index])).indices
          x_subregion[top_indices] = 1
          x_prime[subregion]  = x_subregion

      return x_prime, subregion_index
    

    def pattern_complete(self, region, h_0=None, h_conditioned=None, subregion_index=None, sleep=False, num_iterations=None, sparsity=None):
        num_iterations = num_iterations  if num_iterations != None else getattr(self, region + '_pattern_complete_iterations')
        h = h_0 if h_0 is not None else getattr(self, region)
        w = getattr(self, region + '_' + region)
        for iteration in range(num_iterations):
            h, subregion_index = self.activation(F.linear(h, w), region, h_conditioned, subregion_index, sleep=sleep, sparsity=sparsity)
        return h
    

    def mtl_generate(self, semantic_charge, num_iterations=None):
        num_iterations = num_iterations  if num_iterations != None else getattr(self, 'mtl_generate_pattern_complete_iterations')
        mtl_sparse_sparsity = (semantic_charge/self.max_semantic_charge_input)*self.mtl_sparse_sparsity.clone()
        h_random_sparse = torch.randn(self.mtl_sparse_size)
        h_sparse = self.pattern_complete('mtl_sparse', h_0=h_random_sparse, num_iterations=num_iterations, sparsity=mtl_sparse_sparsity)
        mtl_sparsity = (semantic_charge/self.max_semantic_charge_input)*self.mtl_sparsity.clone()
        h_conditioned = torch.zeros(self.mtl_size)
        h_conditioned[self.mtl_dense_size:] = h_sparse
        h_random = torch.randn(self.mtl_size)
        h_random[self.mtl_dense_size:] = h_sparse
        h = self.pattern_complete('mtl', h_0=h_conditioned, h_conditioned=None, num_iterations=num_iterations, sparsity=mtl_sparsity)
        return h


    def hebbian(self, post_region, pre_region, sleep=False, quick=False, inh=False):
        if self.frozen:
           pass
        else:
          w_name = post_region + '_' + pre_region if not quick else post_region + '_' + pre_region + '_quick'
          w_name = w_name + '_inh' if inh else w_name

          w_name_plastic = w_name + '_plastic'
          w_name_fixed = w_name + '_fixed'
          w_plastic =  getattr(self, w_name_plastic)
          w_fixed = getattr(self, w_name_fixed)
          lmbda = getattr(self, w_name + '_lmbda') if not sleep else getattr(self, post_region + '_' + pre_region + '_lmbda_sleep')
          if w_name == 'ctx_mtl':
            max_post_connectivity = getattr(self, 'max_post_' + w_name)
            total_post_connectivity = torch.sum(w_plastic, dim=1)
            lmbda = (max_post_connectivity + lmbda - total_post_connectivity)
            lmbda = lmbda[:, None]
            self.ctx_mtl_fixed[getattr(self, post_region) == 1, :] = 0

          if w_name == 'ctx_mtl_sparse':
            max_post_connectivity = getattr(self, 'max_post_' + w_name)
            total_post_connectivity = torch.sum(w_plastic, dim=1)
            lmbda = (max_post_connectivity + lmbda - total_post_connectivity)
            lmbda = lmbda[:, None]
            self.ctx_mtl_sparse_fixed[getattr(self, post_region) == 1, :] = 0

          if w_name == 'mtl_sparse_ctx':
            max_post_connectivity = getattr(self, 'max_post_' + w_name)
            total_post_connectivity = torch.sum(w_plastic, dim=1)
            lmbda = (max_post_connectivity + lmbda - total_post_connectivity)
            lmbda = lmbda[:, None]
            self.mtl_sparse_ctx_fixed[getattr(self, post_region) == 1, :] = 0

          if w_name == 'ctx_ctx':

            max_post_connectivity = getattr(self, 'max_post_ctx_mtl')
            total_post_connectivity = torch.sum(self.ctx_mtl_plastic, dim=1)
            
            lmbda = (lmbda*total_post_connectivity/max_post_connectivity)
            lmbda = torch.outer(lmbda, lmbda).sqrt()
            
          if w_name == 'mtl_mtl':
            if self.hebbian_filter:
              lmbda = lmbda*self.mtl_mtl_hebbian_filter


          if inh:
            delta_w = torch.outer(1 - getattr(self, post_region), getattr(self, pre_region))
          else:
            delta_w = torch.outer(getattr(self, post_region), getattr(self, pre_region))

          w_plastic += lmbda*delta_w
          setattr(self, w_name_plastic, w_plastic)
          setattr(self, w_name, w_fixed + w_plastic)


    def homeostasis(self, post_region, pre_region, quick=False, inh=False):
        if self.frozen:
            pass
        else:
          w_name = post_region + '_' + pre_region if not quick else  post_region + '_' + pre_region + '_quick'
          w_name = w_name + '_inh' if inh else w_name
          w_name_plastic = w_name + '_plastic'
          w_name_fixed = w_name + '_fixed'
          w_plastic =  getattr(self, w_name_plastic)
          w_fixed = getattr(self, w_name_fixed)

          max_post_connectivity = getattr(self, 'max_post_' + w_name)
          total_post_connectivity = torch.sum(w_plastic, dim=1)
          post_exceeding_mask = total_post_connectivity > max_post_connectivity

          post_scaling_factors = torch.where(
              post_exceeding_mask,
              max_post_connectivity / total_post_connectivity,
              torch.ones_like(total_post_connectivity)
          )

          w_plastic *= post_scaling_factors.unsqueeze(1)
          setattr(self, w_name_plastic, w_plastic)
          setattr(self, w_name, w_fixed + w_plastic)

          w_plastic = getattr(self, w_name_plastic)
          max_pre_connectivity = getattr(self, 'max_pre_' + w_name)
          total_pre_connectivity = torch.sum(w_plastic, dim=0)
          pre_exceeding_mask = total_pre_connectivity > max_pre_connectivity
          pre_scaling_factors = torch.where(
                pre_exceeding_mask,
                max_pre_connectivity / total_pre_connectivity,
                torch.ones_like(total_pre_connectivity)
            )

          w_plastic *= pre_scaling_factors
          setattr(self, w_name_plastic, w_plastic)
          setattr(self, w_name, w_fixed + w_plastic)


    def init_recordings(self, rec_params):
      self.activity_recordings = {}
      for region in rec_params["regions"]:
        self.activity_recordings[region] = [getattr(self, region)]
      self.activity_recordings_rate = rec_params["rate_activity"]
      self.activity_recordings_time = []
      self.connectivity_recordings = {}
      for connection in rec_params["connections"]:
        self.connectivity_recordings[connection] = [getattr(self, connection)]
      self.connectivity_recordings_time = []
      self.connectivity_recordings_rate = rec_params["rate_connectivity"]

    def record(self):
      if self.time_index%self.activity_recordings_rate == 0:
        for region in self.activity_recordings:
          layer_activity = getattr(self, region)
          self.activity_recordings[region].append(deepcopy(layer_activity.detach().clone()))
          self.activity_recordings_time.append(self.time_index)
      if self.time_index%self.connectivity_recordings_rate == 0:
        for connection in self.connectivity_recordings:
          connection_state = getattr(self, connection)
          self.connectivity_recordings[connection].append(deepcopy(connection_state.detach().clone()))
          self.connectivity_recordings_time.append(self.time_index)

    def init_network(self, net_params):

      #initialize network parameters
      for key, value in net_params.items():
        setattr(self, key, value)

      for region in self.regions:
         num_subregions = getattr(self, region + "_num_subregions")
         size_subregions =  getattr(self, region + "_size_subregions")
         region_size = torch.sum(size_subregions)
         setattr(self, region + "_size", region_size)
         subregions = []
         for subregion_index in range(num_subregions):
            start, end = sum(size_subregions[:subregion_index]), sum(size_subregions[:subregion_index+1])
            subregions.append(torch.arange(start, end))
         setattr(self, region + "_subregions", subregions)
         

      self.frozen = False

      #define subnetworks
      self.sen = torch.zeros((self.sen_size))
      self.mtl_dense_hat = torch.zeros((self.mtl_dense_size))
      self.mtl_sparse_hat = torch.zeros((self.mtl_sparse_size))
      self.mtl_sparse = torch.zeros((self.mtl_sparse_size))
      self.mtl_dense = torch.zeros((self.mtl_dense_size))
      self.mtl_hat = torch.zeros((self.mtl_size))
      self.mtl = torch.zeros((self.mtl_size))
      self.ctx_hat = torch.zeros((self.ctx_size))
      self.ctx = torch.zeros((self.ctx_size))

      #define connectivity


      self.mtl_dense_mtl_dense_plastic = torch.zeros((self.mtl_dense_size, self.mtl_dense_size))
      self.mtl_dense_mtl_dense_fixed = torch.zeros((self.mtl_dense_size, self.mtl_dense_size))
      self.mtl_dense_mtl_dense = self.mtl_dense_mtl_dense_fixed

    
      self.mtl_sparse_mtl_sparse_plastic = torch.zeros((self.mtl_sparse_size, self.mtl_sparse_size))
      self.mtl_sparse_mtl_sparse_fixed = torch.zeros((self.mtl_sparse_size, self.mtl_sparse_size))
      self.mtl_sparse_mtl_sparse = self.mtl_sparse_mtl_sparse_fixed

      self.mtl_mtl_plastic = torch.zeros((self.mtl_size, self.mtl_size))
      self.mtl_mtl_fixed = torch.zeros((self.mtl_size, self.mtl_size))
      self.mtl_mtl = self.mtl_mtl_fixed

      self.ctx_ctx_plastic = torch.zeros((self.ctx_size, self.ctx_size))
      self.ctx_ctx_sparsity_mask = torch.randn((self.ctx_size, self.ctx_size)) < self.ctx_ctx_sparsity
      self.ctx_ctx_fixed = torch.randn((self.ctx_size, self.ctx_size))*self.ctx_ctx_g*self.ctx_ctx_sparsity_mask
      self.ctx_ctx = self.ctx_ctx_fixed


      self.ctx_ctx_inh_plastic = torch.zeros((self.ctx_size, self.ctx_size))
      self.ctx_ctx_inh_fixed = torch.zeros((self.ctx_size, self.ctx_size))
      self.ctx_ctx_inh = self.ctx_ctx_inh_fixed


      if self.mtl_dense_sen_projection:
        self.mtl_dense_sen_plastic = torch.zeros((self.mtl_dense_size, self.sen_size))
        self.mtl_sparse_ctx_fixed = torch.zeros((self.mtl_dense_size, self.sen_size))
        for post_neuron in range(self.mtl_dense_size):
          self.mtl_dense_sen_plastic[post_neuron, torch.randperm(self.sen_size)[:self.mtl_dense_sen_size]] = self.max_post_mtl_dense_sen/self.mtl_dense_sen_size
        self.mtl_dense_sen = self.mtl_dense_sen_plastic.clone()

      else:
        self.mtl_dense_sen = torch.eye(self.sen_size)


      self.ctx_mtl_plastic = torch.zeros((self.ctx_size, self.mtl_size))
      self.ctx_mtl_fixed = self.ctx_mtl_mean + self.ctx_mtl_std*torch.randn((self.ctx_size, self.mtl_size))
      self.ctx_mtl = self.ctx_mtl_fixed + self.ctx_mtl_plastic

      self.ctx_mtl_sparse_plastic = torch.zeros((self.ctx_size, self.mtl_sparse_size))
      self.ctx_mtl_sparse_fixed = self.ctx_mtl_sparse_mean + self.ctx_mtl_sparse_std*torch.randn((self.ctx_size, self.mtl_sparse_size))
      self.ctx_mtl_sparse = self.ctx_mtl_sparse_fixed + self.ctx_mtl_sparse_plastic
    
      self.mtl_sparse_ctx_plastic = torch.zeros((self.mtl_sparse_size, self.ctx_size))
      self.mtl_sparse_ctx_fixed = self.mtl_sparse_ctx_mean + self.mtl_sparse_ctx_std*torch.randn((self.mtl_sparse_size, self.ctx_size))
      self.mtl_sparse_ctx = self.mtl_sparse_ctx_plastic

      self.ctx_mtl_quick_plastic = torch.zeros(self.ctx_size, self.mtl_size)
      self.ctx_mtl_quick_fixed = torch.zeros(self.ctx_size, self.mtl_size)
      self.ctx_mtl_quick = torch.zeros(self.ctx_size, self.mtl_size)



      if self.hebbian_filter:
        self.mtl_mtl_hebbian_filter = self.init_distance_filter(np.inf)



      #initialize temporal variables
      self.day = 0
      self.time_index = 0
      self.awake_indices = []
      self.sleep_indices_A = []
      self.sleep_indices_B = []



    def init_distance_filter(self, radius):
      # Image size
      width, height = 28, 28

      # Define the pixel grid
      x, y = np.meshgrid(np.arange(width), np.arange(height))

      # Flatten the grid to create (784, 2) coordinates
      coords = np.stack([x.ravel(), y.ravel()], axis=1)

      # Compute the pairwise Euclidean distances
      distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))

      # Compute alpha using the equation alpha = -ln(0.01) / r
      alpha = -np.log(0.01) / radius

      # Apply the exponential decay function
      distance_tensor = np.exp(-alpha * distances)

      # The distance_tensor now has a shape of (784, 784)

      hebb_mask = torch.ones((self.mtl_size, self.mtl_size))
      hebb_mask[:self.mtl_dense_size][:, :self.mtl_dense_size] = torch.tensor(distance_tensor)

      return hebb_mask