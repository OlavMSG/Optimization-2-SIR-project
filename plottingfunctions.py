# -*- coding: utf-8 -*-
"""
Created on 18.04.2021

@author: Olav Milian
"""

"""import """
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from SIR_optimal_control_gradient_method import L2norm


"""For nicer plotting"""
sym.init_printing()

fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)


def plotSIRu(tvec, y, u, savename, save=False, pdf=True):
    """
    Function to plot the state variables S, I and R plus the vaccination strategy u

    Parameters
    ----------
    tvec : numpy.ndarray
        timestamp vector.
    y : numpy.ndarray
        3d - vector of timestamps for the state variables S, I and R.
    u : numpy.ndarray
        vector of timestamps for the vaccination strategy.
    savename : string
        name to save the plot as.
    save : bool, optional
        save the plots. The default is False.
    pdf : bool, optional
        if save=True save the plots as pdf. The default is True.

    Returns
    -------
    None.

    """
    # get S, I and R
    S = y[0, :]
    I = y[1, :]
    R = y[2, :]
    # plot
    fig, ax1 = plt.subplots(1, 1, num=1)
    # add second y-axis
    ax2 = ax1.twinx()
    # plot S, I, R
    ax1.plot(tvec, S, label="$S(t)$, susceptible", color="c")
    ax1.plot(tvec, I, label="$I(t)$, infected", color="m")
    ax1.plot(tvec, R, label="$R(t)$, susceptible", color="g")
    # plot u
    ax2.plot(tvec, u, label="$u(t)$, vaccination strategy", color="k")
    ax2.plot(tvec, 0.9*np.ones_like(u), "k--", label="Vaccination limit")
    # set plot title
    ax1.set_title("SIR optimal control")
    # label axis
    ax1.set_ylabel("#People")
    ax1.set_xlabel("$t$, time")
    ax2.set_ylabel("Part of susceptible population \n that get vaccinated")
    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    # save?
    # as pdf?
    if save and pdf:
        plt.savefig("plot/" + savename + ".pdf", bbox_inches='tight')
    elif save:
        plt.savefig("plot/" + savename + ".png", bbox_inches='tight')
    plt.show()


def plotAdjoint(tvec, p, savename, save=False, pdf=True):
    """
    Function to plot the adjoint state variables

    Parameters
    ----------
    tvec : numpy.ndarray
        timestamp vector.
    p : numpy.ndarray
        3d - vector of timestamps for the adjoint variables p1, p2 and p3.
    savename : string
        name to save the plot as.
    save : bool, optional
        save the plots. The default is False.
    pdf : bool, optional
        if save=True save the plots as pdf. The default is True.

    Returns
    -------
    None.

    """
    # get p1, p2 and p3
    p1 = p[0, :]
    p2 = p[1, :]
    p3 = p[2, :]
    # plot
    fig, ax1 = plt.subplots(1, 1, num=1)
    # plot p1, p2, p3
    ax1.plot(tvec, p1, label="$p_1(t)$", color="c")
    ax1.plot(tvec, p2, label="$p_2(t)$", color="m")
    ax1.plot(tvec, p3, label="$p_3(t)$", color="g")
    # set plot title
    ax1.set_title("Adjoint variables")
    # label axis
    ax1.set_ylabel("Adjoint variables")
    ax1.set_xlabel("$t$, time")
    # legend
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    # save?
    # as pdf?
    if save and pdf:
        plt.savefig("plot/" + savename + ".pdf", bbox_inches='tight')
    elif save:
        plt.savefig("plot/" + savename + ".png", bbox_inches='tight')
    plt.show()


def plotRel(tvec, u_list, fu_list, savename, save=False, pdf=True):
    """
    Function to plot the relative errors for the vaccination strategy and the costfunctional

    Parameters
    ----------
    tvec : numpy.ndarray
        timestamp vector, used for the L2-norm.
    u_list : numpy.ndarray
        vector of vectors of vaccination strategies.
    fu_list : numpy.ndarray
        vector of values for the cost functional.
    savename : string
        name to save the plot as.
    save : bool, optional
        save the plots. The default is False.
    pdf : bool, optional
        if save=True save the plots as pdf. The default is True.
    Returns
    -------
    None.

    """
    # get the relative errors
    fu_rel = np.abs(fu_list[-1] - fu_list) / np.abs(fu_list[-1])
    u_rel = np.asarray([L2norm(u_list[-1] - u, tvec) for u in u_list]) / L2norm(u_list[-1], tvec)
    # plot
    fig, ax1 = plt.subplots(1, 1, num=1)
    # plot S, I, R
    ax1.semilogy(u_rel[:-1], label="$\\frac{||u_{end}-u_k||_{L^2(0,T)}}{||u_{end}||_{L^2(0,T)}}$", color="c")
    ax1.semilogy(fu_rel[:-1], label="$\\frac{|f(u_{end})-f(u_k)|}{|f(u_{end})|}$", color="m")
    # set plot title
    ax1.set_title("Relative Errors")
    # label axis
    ax1.set_xlabel("$k$, iteration number")
    # legend
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    # save?
    # as pdf?
    if save and pdf:
        plt.savefig("plot/" + savename + ".pdf", bbox_inches='tight')
    elif save:
        plt.savefig("plot/" + savename + ".png", bbox_inches='tight')
    plt.show()

