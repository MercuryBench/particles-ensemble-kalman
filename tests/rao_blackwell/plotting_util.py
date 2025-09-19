# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:07:24 2025

@author: pkw32
"""

import corner
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from particles import resampling


def extended_corner(samples, prior, weights=None, truths=None, param_labels=None):
  if param_labels is None:
    param_labels = [p for p in prior.keys()]
  fig_corner = corner.corner(samples.T, truths=truths, labels=param_labels, weights =weights, hist_kwargs={"density": True})
  num_params = samples.shape[0]
  axes = np.array(fig_corner.axes).reshape((num_params, num_params))
  for i, param in enumerate(prior.keys()):
     ax = axes[i, i]
     
     # get the limits of the x-axis
     xlim = ax.get_xlim()
     x = np.linspace(*xlim, 200)
     
     # choose your reference density: here a normal distribution centered at true mean
     ref_pdf = prior[param].pdf(x)
     
     ax.plot(x, ref_pdf, "--", color="red", lw=1, label="prior")
     ax.legend()

def ABP_plot_thetaPart(pf, wgts, xs=None, ys=None, k=None, N_smp=50, n_vis_list=None):
    angles = pf.X.w
    ms = [MC.mean for MC in pf.X.MC]
    Cs = [MC.cov for MC in pf.X.MC]
    
    plt.figure(figsize=(9,6))
    plt.subplot(231)
    plt.hist(np.mod(angles, 2*pi),50,weights=wgts.W, density=True)
    if k is not None:
        plt.axvline(np.mod(xs['w'][k], 2*pi), color='tab:orange', label="true angle")
        #plt.axvline(angles[k], color='k')
    plt.xlim([-0.5,2*pi+0.5])
    plt.title("angle")
    plt.xlabel("$\\phi$")
    plt.legend()
    
    indices = resampling.stratified(pf.wgts.W)
    
    samples = np.vstack([np.random.multivariate_normal(ms[k].flatten(), Cs[k], N_smp) for k in indices])
    
    plt.subplot(234)
    # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
    plt.hist2d(samples[:,0], samples[:, 1], 20, cmap="viridis_r")
    if k is not None:
        plt.plot(xs['z'][k][0], xs['z'][k][1], '.', color="tab:orange", markersize=8, label="true $x$, $y$")
        plt.plot(ys[k][0], ys[k][1], '.', color="tab:green", markersize=8, label="observed $x$, $y$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x,y")
    plt.legend(loc="lower right")
    
    plt.subplot(235)
    plt.plot(xs['w'], color="tab:orange", label="true angle")
    vp = plt.violinplot([pf.hist.X[k].w for k in range(len(ys))], positions = range(len(ys)), widths=1, showextrema=False)
    plt.xlabel("iteration")
    plt.ylabel("$\\phi$")
    plt.legend()
    # Change face color
    for body in vp['bodies']:
        body.set_facecolor('tab:blue')
    plt.title("angle")
    
    plt.subplot(232)
    plt.hist(samples[:,2], 50, density=True)
    if k is not None:
        plt.axvline(xs['z'][k,2], color='tab:orange', label="true speed")
    plt.title("speed")
    plt.xlabel("v")
    plt.legend()
    # plt.xlim([-30,30])
    
    plt.subplot(233)
    angles_resampled_padded = np.concatenate([np.array([angles[indices[k]] for n in range(N_smp)]) for k in range(len(wgts.W)) ])
    plt.hist2d(angles_resampled_padded, samples[:,2], 30, cmap="viridis_r")
    plt.plot(xs['w'][k], xs['z'][k,2], '.', color="tab:orange", markersize=8, label="true $\\phi$, $v$")
    plt.xlabel("$\\phi$")
    plt.ylabel("v")
    plt.title("angle, speed")
    plt.legend()
    
    plt.subplot(236)
    plt.plot(xs['z'][:,0], xs['z'][:,1], '.-', color="tab:orange", label="state")
    plt.plot(ys[:,0], ys[:,1], '.-', label="data", color="tab:green")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("state estimation")
    # n_vis = len(pf.hist.X)//3
    if n_vis_list is None:
      n_vis_list = [0, -1]
    for n_vis in n_vis_list:
      indices_t = resampling.stratified(pf.hist.wgts[n_vis].W)
      ms_t = [MC.mean for MC in pf.hist.X[n_vis].MC]
      Cs_t = [MC.cov for MC in pf.hist.X[n_vis].MC]
      samples_t = np.vstack([np.random.multivariate_normal(ms_t[k].flatten(), Cs_t[k], 10) for k in indices_t])
      mean_t = np.mean(samples_t, axis=0)[0:2]
      cov_t = np.cov(samples_t.T)[0:2,0:2]
      evs, evecs = np.linalg.eig(cov_t)
      angle_ellipse = np.arctan2(evecs[1,0], evecs[0,0])
      from matplotlib.patches import Ellipse
      
      ellipse = Ellipse(xy = (mean_t[0], mean_t[1]), width =3*np.sqrt(evs[0]), height = 3*np.sqrt(evs[1]), angle=angle_ellipse, edgecolor='tab:blue', facecolor=None, label="filter" if n_vis==0 else None)
      
      # plt.plot(samples_t[:,0],samples_t[:,1], '.k', alpha=0.01)
      ax = plt.gca()
      ax.add_patch(ellipse)
    
    plt.legend()
    plt.tight_layout()
    
def ABP_plot_thetaPart_small(pf, wgts, xs=None, ys=None, k=None, N_smp=50, n_vis_list=None):
    angles = pf.X.w
    ms = [MC.mean for MC in pf.X.MC]
    Cs = [MC.cov for MC in pf.X.MC]
    N = len(pf.hist.wgts)
    
    plt.figure(figsize=(9,6))
    # plt.subplot(223)
    # plt.hist(np.mod(angles, 2*pi),50,weights=wgts.W, density=True)
    # if k is not None:
    #     plt.axvline(np.mod(xs['w'][k], 2*pi), color='tab:orange', label="true angle")
    #     #plt.axvline(angles[k], color='k')
    # plt.xlim([-0.5,2*pi+0.5])
    # plt.title("angle")
    # plt.xlabel("$\\phi$")
    # plt.legend()
    
    indices = resampling.stratified(pf.wgts.W)
    
    samples = np.vstack([np.random.multivariate_normal(ms[k].flatten(), Cs[k], N_smp) for k in indices])
    means = np.zeros((N,2))
    covs = np.zeros((N,2,2))
    for n_vis in range(N):
      indices_t = resampling.stratified(pf.hist.wgts[n_vis].W)
      ms_t = [MC.mean for MC in pf.hist.X[n_vis].MC]
      Cs_t = [MC.cov for MC in pf.hist.X[n_vis].MC]
      samples_t = np.vstack([np.random.multivariate_normal(ms_t[k].flatten(), Cs_t[k], 10) for k in indices_t])
      means[n_vis, :] = np.mean(samples_t, axis=0)[0:2]
      covs[n_vis, :, :] = np.cov(samples_t.T)[0:2,0:2]
    # plt.subplot(234)
    # # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
    # plt.hist2d(samples[:,0], samples[:, 1], 20, cmap="viridis_r")
    # if k is not None:
    #     plt.plot(xs['z'][k][0], xs['z'][k][1], '.', color="tab:orange", markersize=8, label="true $x$, $y$")
    #     plt.plot(ys[k][0], ys[k][1], '.', color="tab:green", markersize=8, label="observed $x$, $y$")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("x,y")
    # plt.legend(loc="lower right")
    
    plt.subplot(221)
    plt.plot(xs['z'][:,0], '.-', color="tab:orange", label="state")
    # plt.plot(ys[:,0], '-', label="data", color="tab:green")
    
    plt.plot(means[:, 0], label="mean")
    # plt.plot(means[:, 0]+2*np.sqrt(covs[:,0,0]), label="mean")
    plt.fill_between(range(N), means[:, 0]+3*np.sqrt(covs[:,0,0]), means[:, 0]-3*np.sqrt(covs[:,0,0]), alpha=0.25)
    # plt.plot(xs['z'][:,0], color="tab:red", label="true $x_1$")
    plt.xlabel("iteration")
    plt.ylabel("$x_1$")
    plt.title("state, 1st component")
    plt.legend()
    
    plt.subplot(222)
    plt.plot(xs['z'][:,1], '.-', color="tab:orange", label="state")
    # plt.plot(ys[:,1], '.-', label="data", color="tab:green")
    
    plt.plot(means[:, 1], label="mean")
    # plt.plot(means[:, 0]+2*np.sqrt(covs[:,0,0]), label="mean")
    plt.fill_between(range(N), means[:, 1]+3*np.sqrt(covs[:,1,1]), means[:, 1]-3*np.sqrt(covs[:,1,1]), alpha=0.25)
    plt.legend()
    
    plt.xlabel("iteration")
    plt.ylabel("$x_2$")
    plt.title("state, 2nd component")
    plt.legend()
    
    plt.subplot(223)
    plt.plot(xs['w'], color="tab:orange", label="true angle")
    vp = plt.violinplot([pf.hist.X[k].w for k in range(len(ys))], positions = range(len(ys)), widths=1, showextrema=False)
    plt.xlabel("iteration")
    plt.ylabel("$\\phi$")
    plt.legend()
    # Change face color
    for body in vp['bodies']:
        body.set_facecolor('tab:blue')
    plt.title("angle")
    
    # plt.subplot(232)
    # plt.hist(samples[:,2], 50, density=True)
    # if k is not None:
    #     plt.axvline(xs['z'][k,2], color='tab:orange', label="true speed")
    # plt.title("speed")
    # plt.xlabel("v")
    # plt.legend()
    # # plt.xlim([-30,30])
    
    # plt.subplot(233)
    # angles_resampled_padded = np.concatenate([np.array([angles[indices[k]] for n in range(N_smp)]) for k in range(len(wgts.W)) ])
    # plt.hist2d(angles_resampled_padded, samples[:,2], 30, cmap="viridis_r")
    # plt.plot(xs['w'][k], xs['z'][k,2], '.', color="tab:orange", markersize=8, label="true $\\phi$, $v$")
    # plt.xlabel("$\\phi$")
    # plt.ylabel("v")
    # plt.title("angle, speed")
    # plt.legend()
    
    plt.subplot(224)
    plt.plot(xs['z'][:,0], xs['z'][:,1], '.-', color="tab:orange", label="state")
    plt.plot(ys[:,0], ys[:,1], '.-', label="data", color="tab:green")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("state estimation")
    # n_vis = len(pf.hist.X)//3
    if n_vis_list is None:
      n_vis_list = [0, -1]
    for n_vis in n_vis_list:
      indices_t = resampling.stratified(pf.hist.wgts[n_vis].W)
      ms_t = [MC.mean for MC in pf.hist.X[n_vis].MC]
      Cs_t = [MC.cov for MC in pf.hist.X[n_vis].MC]
      samples_t = np.vstack([np.random.multivariate_normal(ms_t[k].flatten(), Cs_t[k], 10) for k in indices_t])
      mean_t = np.mean(samples_t, axis=0)[0:2]
      cov_t = np.cov(samples_t.T)[0:2,0:2]
      evs, evecs = np.linalg.eig(cov_t)
      angle_ellipse = np.arctan2(evecs[1,0], evecs[0,0])
      from matplotlib.patches import Ellipse
      
      ellipse = Ellipse(xy = (mean_t[0], mean_t[1]), width =3*np.sqrt(evs[0]), height = 3*np.sqrt(evs[1]), angle=angle_ellipse, edgecolor='tab:blue', facecolor=None, label="filter" if n_vis==0 else None)
      
      # plt.plot(samples_t[:,0],samples_t[:,1], '.k', alpha=0.01)
      ax = plt.gca()
      ax.add_patch(ellipse)
    
    plt.legend()
    plt.tight_layout()


def ABP_plot_thetaPart_final(pf, wgts, xs=None, ys=None, k=None, N_smp=50, n_vis_list=None):
    angles = pf.X.w
    ms = [MC.mean for MC in pf.X.MC]
    Cs = [MC.cov for MC in pf.X.MC]
    
    plt.figure(figsize=(9,6))
    plt.subplot(221)
    plt.hist(np.mod(angles, 2*pi),50,weights=wgts.W, density=True)
    if k is not None:
        plt.axvline(np.mod(xs['w'][k], 2*pi), color='tab:orange', label="true angle")
        #plt.axvline(angles[k], color='k')
    plt.xlim([-0.5,2*pi+0.5])
    plt.title("angle")
    plt.xlabel("$\\phi$")
    plt.legend()
    
    indices = resampling.stratified(pf.wgts.W)
    
    samples = np.vstack([np.random.multivariate_normal(ms[k].flatten(), Cs[k], N_smp) for k in indices])
    
    plt.subplot(224)
    # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
    plt.hist2d(samples[:,0], samples[:, 1], 20, cmap="viridis_r")
    if k is not None:
        plt.plot(xs['z'][k][0], xs['z'][k][1], '.', color="tab:orange", markersize=8, label="true $x$, $y$")
        plt.plot(ys[k][0], ys[k][1], '.', color="tab:green", markersize=8, label="observed $x$, $y$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x,y")
    plt.legend(loc="lower right")
    
    
    plt.subplot(222)
    plt.hist(samples[:,2], 50, density=True)
    if k is not None:
        plt.axvline(xs['z'][k,2], color='tab:orange', label="true speed")
    plt.title("speed")
    plt.xlabel("v")
    plt.legend()
    # plt.xlim([-30,30])
    
    plt.subplot(223)
    angles_resampled_padded = np.concatenate([np.array([angles[indices[k]] for n in range(N_smp)]) for k in range(len(wgts.W)) ])
    plt.hist2d(angles_resampled_padded, samples[:,2], 30, cmap="viridis_r")
    plt.plot(xs['w'][k], xs['z'][k,2], '.', color="tab:orange", markersize=8, label="true $\\phi$, $v$")
    plt.xlabel("$\\phi$")
    plt.ylabel("v")
    plt.title("angle, speed")
    plt.legend()
    plt.tight_layout()
    
    