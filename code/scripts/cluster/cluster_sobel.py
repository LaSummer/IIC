from __future__ import print_function

import argparse
import itertools
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch
import faiss
import torch.nn as nn

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import code.archs as archs
from code.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from code.utils.cluster.transforms import sobel_process
from code.utils.cluster.data import cluster_create_dataloaders
from code.utils.cluster.IID_losses import IID_loss
from code.utils.cluster.cluster_eval import cluster_eval
import code.utils.cluster.kmeans_clustering as kclustering
"""
  Semisupervised overclustering ("IIC+" = "IID+")
  Note network is trained entirely unsupervised, as labels are found for 
  evaluation only and do not affect the network.
  Train and test script (coloured datasets).
  Network has one output head only.
"""

# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--opt", type=str, default="Adam")
parser.add_argument("--mode", type=str, default="IID+")

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)

parser.add_argument("--gt_k", type=int, required=True)
parser.add_argument("--output_k", type=int, required=True)
parser.add_argument("--lamb", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")
parser.add_argument("--restart", default=False, action="store_true")
parser.add_argument("--restart_from_best", dest="restart_from_best",
                    default=False, action="store_true")
parser.add_argument("--test_code", default=False, action="store_true")

parser.add_argument("--save_freq", type=int, default=10)

parser.add_argument("--batchnorm_track", default=False, action="store_true")

# transforms
parser.add_argument("--mix_train", default=False, action="store_true")
parser.add_argument("--include_rgb", default=False, action="store_true")
parser.add_argument("--demean", default=False, action="store_true")
parser.add_argument("--per_img_demean", dest="per_img_demean", default=False,
                    action="store_true")
parser.add_argument("--data_mean", type=float, nargs="+", default=[])
parser.add_argument("--data_std", type=float, nargs="+", default=[])

parser.add_argument("--crop_orig", default=False, action="store_true")
parser.add_argument("--rand_crop_sz", type=int, default=84)
parser.add_argument("--input_sz", type=int, default=96)
parser.add_argument("--fluid_warp", default=False, action="store_true")
parser.add_argument("--rand_crop_szs_tf", type=int, nargs="+",
                    default=[])  # only used if fluid warp true
parser.add_argument("--rot_val", type=float,
                    default=0.)  # only used if fluid warp true

parser.add_argument("--cutout", default=False, action="store_true")
parser.add_argument("--cutout_p", type=float, default=0.5)
parser.add_argument("--cutout_max_box", type=float, default=0.5)

config = parser.parse_args()

# Setup ------------------------------------------------------------------------

config.twohead = False
if not config.include_rgb:
  config.in_channels = 2
else:
  config.in_channels = 5

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)

assert (config.mode == "IID+")
assert (config.output_k >= config.gt_k)
config.eval_mode = "orig"
config.double_eval = False

if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

if config.restart:
  config_name = "config.pickle"
  net_name = "latest_net.pytorch"
  opt_name = "latest_optimiser.pytorch"

  if config.restart_from_best:
    config_name = "best_config.pickle"
    net_name = "best_net.pytorch"
    opt_name = "best_optimiser.pytorch"

  given_config = config
  reloaded_config_path = os.path.join(given_config.out_dir, config_name)
  print("Loading restarting config from: %s" % reloaded_config_path)
  with open(reloaded_config_path, "rb") as config_f:
    config = pickle.load(config_f)
  assert (config.model_ind == given_config.model_ind)
  config.restart = True
  config.restart_from_best = given_config.restart_from_best

  # copy over new num_epochs and lr schedule
  config.num_epochs = given_config.num_epochs
  config.lr_schedule = given_config.lr_schedule

else:
  print("Config: %s" % config_to_str(config))

# Model ------------------------------------------------------------------------

dataloaders, dataset_imgs, flatten_dataset_imgs, flatten_dataloaders, mapping_assignment_dataloader, mapping_test_dataloader = \
  cluster_create_dataloaders(config)

net = archs.__dict__[config.arch](config)
if config.restart:
  model_path = os.path.join(config.out_dir, net_name)
  net.load_state_dict(
    torch.load(model_path, map_location=lambda storage, loc: storage))
net.cuda()
net = torch.nn.DataParallel(net)
net.train()

optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
if config.restart:
  optimiser.load_state_dict(
    torch.load(os.path.join(config.out_dir, opt_name)))
  
kmeans_crit = nn.CrossEntropyLoss().cuda()

# Results ----------------------------------------------------------------------

if config.restart:
  if not config.restart_from_best:
    next_epoch = config.last_epoch + 1  # corresponds to last saved model
  else:
    # sanity check
    next_epoch = np.argmax(np.array(config.epoch_acc)) + 1
    assert (next_epoch == config.last_epoch + 1)
  print("starting from epoch %d" % next_epoch)

  config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
  config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
  config.epoch_stats = config.epoch_stats[:next_epoch]

  config.epoch_loss = config.epoch_loss[:(next_epoch - 1)]
  config.epoch_loss_no_lamb = config.epoch_loss_no_lamb[:(next_epoch - 1)]
else:
  config.epoch_acc = []
  config.epoch_avg_subhead_acc = []
  config.epoch_stats = []

  config.epoch_loss = []
  config.epoch_loss_no_lamb = []

  _ = cluster_eval(config, net,
                   mapping_assignment_dataloader=mapping_assignment_dataloader,
                   mapping_test_dataloader=mapping_test_dataloader,
                   sobel=True)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  sys.stdout.flush()
  next_epoch = 1

fig, axarr = plt.subplots(4, sharex=False, figsize=(20, 20))

# Train ------------------------------------------------------------------------
deepcluster = kclustering.Kmeans(config.output_k)

def reconstruct(train_dataset):
  # return train_dataset
  new_dataloaders = []
  for _, dataset in enumerate(train_dataset):
    new_dataloaders.append(
      torch.utils.data.DataLoader(
        dataset,
        batch_size=config.dataloader_batch_sz,
        shuffle=False,
        num_workers=0,
        drop_last=False
      )
    )
  return new_dataloaders

def reconstruct_v2(train_dataset):
  new_dataloaders = []
  n = config.num_dataloaders + 1
  subset_size = len(train_dataset) / n
  for i in range(n):
    indexes = list(range(i*subset_size, (i+1)*subset_size))
    subset = torch.utils.data.Subset(train_dataset, indexes)
    new_dataloaders.append(torch.utils.data.DataLoader(
      subset,
      batch_size=config.dataloader_batch_sz,
      # shuffle=shuffle,
      # num_workers=0,
      # drop_last=False
    ))
  return new_dataloaders

def compute_features(dataloaders, net, N):
    net.eval()
    iterators = (d for d in dataloaders)
    i = 0
    print("LENGTH OF DATASET: ", N)
    print("LENGTH OF DATALOADER0: ", len(dataloaders[0]))
    #print("LENGTH OF DATALOADER1: ", len(dataloaders[1]))
    #print("LENGTH OF DATALOADER2: ", len(dataloaders[2]))
    for tup in itertools.izip(*iterators):
      imgs_curr = tup[0][0].cuda()  # always the first
      #imgs_tf_1 = tup[1][0].cuda()
      #imgs_tf_2 = tup[2][0].cuda()
      curr_batch_sz = imgs_curr.size(0)
      #curr_batch_sz1 = imgs_tf_1.size(0)
      #curr_batch_sz2 = imgs_tf_2.size(0)
      if i % 10 == 0:
        print("ith features input batch_sz: " + str(i) + ' ' + str(curr_batch_sz))
      imgs_curr = sobel_process(imgs_curr, config.include_rgb)
      #imgs_tf_1 = sobel_process(imgs_tf_1, config.include_rgb)
      #imgs_tf_2 = sobel_process(imgs_tf_2, config.include_rgb)

      x_outs = net(imgs_curr, kmeans_use_features=True,)[0].data.cpu().numpy()
      #x_tf_1_outs = net(imgs_tf_1, kmeans_use_features=True,)[0].data.cpu().numpy()
      #x_tf_2_outs = net(imgs_tf_1, kmeans_use_features=True,)[0].data.cpu().numpy()
      if i == 0:
        features1 = np.zeros((N/3, x_outs.shape[1]), dtype='float32')
        #features2 = np.zeros((N/3, x_tf_1_outs.shape[1]), dtype='float32')
        #features3 = np.zeros((N/3, x_tf_2_outs.shape[1]), dtype='float32')
      x_outs = x_outs.astype('float32')
      #x_tf_1_outs = x_tf_1_outs.astype('float32')
      #x_tf_2_outs = x_tf_2_outs.astype('float32')
      if curr_batch_sz == config.dataloader_batch_sz:
        features1[i * curr_batch_sz: (i + 1) * curr_batch_sz] = x_outs
        #features2[i * curr_batch_sz: (i + 1) * curr_batch_sz] = x_tf_1_outs
        #features3[i * curr_batch_sz: (i + 1) * curr_batch_sz] = x_tf_2_outs
      else:
        # special treatment for final batch
        features1[i * curr_batch_sz:] = x_outs
        #features2[i * curr_batch_sz:] = x_tf_1_outs
        #features3[i * curr_batch_sz:] = x_tf_2_outs
      i += 1
    
    net.train()
    return features1

for e_i in xrange(next_epoch, config.num_epochs):
  print("Starting e_i: %d" % e_i)
  sys.stdout.flush()

  b_i = 0
  if e_i in config.lr_schedule:
    optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

  avg_loss = 0.
  avg_loss_no_lamb = 0.
  avg_loss_count = 0
  # generate and assign pseudo labels for the whole dataset (including origin and transformed dataset)

  # 1. get kmeans_use_feature:
  print('Start compute features:')
  #features = compute_features(dataloaders, net, 300000) #TODO: change type of dataset_imgs
  features = np.zeros((100000, 512), dtype='float32')
  print("computed feature shape:", features.shape)

  # 2. cluster the features using kmeans
  print('Cluster the kmeans features')
  pseudo_labels = deepcluster.cluster(features, verbose=True)

  # 3. assign pseudo labels
  #print('Assign pseudo labels')
  #train_dataset = kclustering.cluster_assign(deepcluster.images_lists, dataset_imgs) #TODO: change type of dataset_imgs

  #print("start reconstruct dataset:")
  #dataloaders = reconstruct([train_dataset, dataset_imgs[1], dataset_imgs[2]])

  iterators = (d for d in dataloaders)
  
  itern = 0
  for tup in itertools.izip(*iterators):
    itern += 1
    net.module.zero_grad()

    # one less because this is before sobel
    all_imgs = torch.zeros(config.batch_sz, config.in_channels - 1,
                           config.input_sz,
                           config.input_sz).cuda()
    all_imgs_tf = torch.zeros(config.batch_sz, config.in_channels - 1,
                              config.input_sz,
                              config.input_sz).cuda()
    # pseudo labels from dc

    imgs_curr = tup[0][0]  # always the first
    #imgs_curr_target = tup[0][1]

    if itern == 1:
      print("varyfing tup[0][0] shape:", tup[0][0].size())
      print("varyfing tup[0][1] shape:", tup[0][1].size())

    curr_batch_sz = imgs_curr.size(0)
    for d_i in xrange(config.num_dataloaders):
      imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
      assert (curr_batch_sz == imgs_tf_curr.size(0))

      actual_batch_start = d_i * curr_batch_sz
      actual_batch_end = actual_batch_start + curr_batch_sz
      all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
        imgs_curr.cuda()
      all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
        imgs_tf_curr.cuda()

    if not (curr_batch_sz == config.dataloader_batch_sz):
      print("last batch sz %d" % curr_batch_sz)

    curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
    all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
    all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]
    #all_imgs_target = torch.cat((imgs_curr_target, imgs_curr_target), -1)
    #get pseudo labels
    imgs_curr_target = torch.IntTensor(pseudo_labels[(itern-1)*config.dataloader_batch_sz:itern*config.dataloader_batch_sz])
    all_imgs_target = torch.cat((imgs_curr_target, imgs_curr_target), -1)
    if itern == 1:
      print("varyfing all_imgs_target shape:", all_imgs_target.size())

    all_imgs = sobel_process(all_imgs, config.include_rgb)
    all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

    x_outs = net(all_imgs)
    x_tf_outs = net(all_imgs_tf)

    avg_loss_batch = None  # avg over the heads
    avg_loss_no_lamb_batch = None
    avg_kmeans_loss = None 
    for i in xrange(config.num_sub_heads):
      loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i], lamb=config.lamb)
      if itern == 1 and i == 0:
        print("varifying x_out[0] size:", x_outs[i].size())
        print("varifying all_imgs_target size", all_imgs_target.long().size())
      kloss = kmeans_crit(x_outs[i], all_imgs_target.long())
      if avg_loss_batch is None:
        avg_loss_batch = loss
        avg_loss_no_lamb_batch = loss_no_lamb
        avg_kmeans_loss = avg_kmeans_loss
      else:
        avg_loss_batch += loss
        avg_loss_no_lamb_batch += loss_no_lamb
        avg_kmeans_loss += avg_kmeans_loss

    avg_loss_batch /= config.num_sub_heads
    avg_loss_no_lamb_batch /= config.num_sub_heads
    avg_kmeans_loss /= config.num_sub_heads

    if ((b_i % 100) == 0) or (e_i == next_epoch):
      print("Model ind %d epoch %d batch: %d avg loss %f avg loss no lamb %f avg kloss %f"
            "time %s" % \
            (config.model_ind, e_i, b_i, avg_loss_batch.item(),
             avg_loss_no_lamb_batch.item(), avg_kmeans_loss.item(), datetime.now()))
      sys.stdout.flush()

    if not np.isfinite(avg_loss_batch.item()):
      print("Loss is not finite... %s:" % str(avg_loss_batch))
      exit(1)

    avg_loss += avg_loss_batch.item()
    avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
    avg_loss_count += 1

    avg_loss_batch.backward()

    optimiser.step()

    b_i += 1
    if b_i == 2 and config.test_code:
      break

  # Eval -----------------------------------------------------------------------

  avg_loss = float(avg_loss / avg_loss_count)
  avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

  config.epoch_loss.append(avg_loss)
  config.epoch_loss_no_lamb.append(avg_loss_no_lamb)

  is_best = cluster_eval(config, net,
                         mapping_assignment_dataloader=mapping_assignment_dataloader,
                         mapping_test_dataloader=mapping_test_dataloader,
                         sobel=True)

  print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
  sys.stdout.flush()

  axarr[0].clear()
  axarr[0].plot(config.epoch_acc)
  axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

  axarr[1].clear()
  axarr[1].plot(config.epoch_avg_subhead_acc)
  axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

  axarr[2].clear()
  axarr[2].plot(config.epoch_loss)
  axarr[2].set_title("Loss")

  axarr[3].clear()
  axarr[3].plot(config.epoch_loss_no_lamb)
  axarr[3].set_title("Loss no lamb")

  fig.tight_layout()
  fig.canvas.draw_idle()
  fig.savefig(os.path.join(config.out_dir, "plots.png"))

  if is_best or (e_i % config.save_freq == 0):
    net.module.cpu()

    if e_i % config.save_freq == 0:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "latest_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "latest_optimiser.pytorch"))
      config.last_epoch = e_i  # for last saved version

    if is_best:
      torch.save(net.module.state_dict(),
                 os.path.join(config.out_dir, "best_net.pytorch"))
      torch.save(optimiser.state_dict(),
                 os.path.join(config.out_dir, "best_optimiser.pytorch"))

      with open(os.path.join(config.out_dir, "best_config.pickle"),
                'wb') as outfile:
        pickle.dump(config, outfile)

      with open(os.path.join(config.out_dir, "best_config.txt"),
                "w") as text_file:
        text_file.write("%s" % config)

    net.module.cuda()

  with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
    pickle.dump(config, outfile)

  with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
    text_file.write("%s" % config)

  if config.test_code:
    exit(0)



      


        



