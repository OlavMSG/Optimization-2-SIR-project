# -*- coding: utf-8 -*-
"""
Created on 25.03.2021

@author: Olav Milian
"""

import numpy as np


def L2norm(v, tvec):
    """
    Function to compute the L2-norm by the trapezoidal method for a function given the timestamps

    Parameters
    ----------
    v : numpy.ndarray
        vector of timestamps for the function.
    tvec : numpy.ndarray
        vector of timestamps.

    Returns
    -------
    float
        The L2-norm of the function v.

    """
    return np.sqrt(np.trapz(v ** 2, tvec))


def ProjectedGradientMethod(CP, tol, maxiter=100, maxiter2=50, alpha0=1.0, rho=0.1, c=1e-4, alpha_tol=1e-10, Nq=4):
    """
    Function to preform a Projected gradient method for a controlproblem, 
    using the Nq-th gaussian quadrature-rule to evaluate the costfunctional.

    Parameters
    ----------
    CP : ControlProblem
        ControlProblem to use the gradient method on.
    tol : float
        tolerance for convergence in L2-norm.
    maxiter : int, optional
        maximum number of iterations. The default is 100.
    maxiter2 : int, optional
        maximum number of iterations to find alpha. The default is 50.
        Note: after 11 alpha < alpha_tol if default  parameters are used
    alpha0 : float, optional
        the initial alpha to for each alpha-iteration. The default is 1.0.
    rho : float, optional
        shrinkage factor. The default is 0.1.
    c : float, optional
        the constant in Armijo-backtracking. The default is 1e-4.
    alpha_tol : float, optional
        tolerance for alpha. The default is 1e-10.
    Nq : int, optional
        which quadrature-rule to use in the guassian quadrature for the evaluating the reduced costfunctional. The default is 4.

    Returns
    -------
    y : numpy.ndarray
        3d - vector of timestamps for the state variables S, I and R.
    u : numpy.ndarray
        vector of timestamps for the vaccination strategy.
    p : numpy.ndarray
        3d - vector of timestamps for the adjoint variables p1, p2 and p3.
    iteration : int
        the number of iterations.
    u_list : numpy.ndarray
        vector of vectors of vaccination strategies.
    fu_list : numpy.ndarray
        vector of values for the cost functional.

    """

    # initial setup
    iteration = 0
    u_list = []
    fu_list = []
    u_current = np.zeros(CP.SEq.M + 1)  # u0
    # solve state and Adjoint
    y_current = CP.SEq.RK4(u_current)
    p_current = CP.AEq.RK4(u_current, y_current)
    fu_current = CP.CostFunc.eval_reduced(y_current, u_current, Nq)
    # add current to list
    u_list.append(u_current)
    fu_list.append(fu_current)
    # do iterations
    for i in range(maxiter):
        print("Iteration: ", i + 1)
        # get current search direction
        v_current = - CP.CostFunc.gradient(y_current, p_current, u_current)
        # now find alpha
        # set alpha to alpha0
        alpha_current = alpha0
        # get the current gradient
        g_current = CP.CostFunc.gradient(y_current, p_current, u_current)
        for j in range(maxiter2):
            print("alpha iteration", j + 1, ", alpha =", alpha_current)
            # potential next u
            u_bar = CP.BoxCon.projection(u_current + alpha_current * v_current)
            # try to solve the system, if fail decrease alpha
            force_decrease = False
            try:
                # try to compute y, p and fu from u
                y_bar = CP.SEq.RK4(u_bar)
                p_bar = CP.AEq.RK4(u_bar, y_bar)
                fu_bar = CP.CostFunc.eval_reduced(y_bar, u_bar, Nq)
            except ValueError:
                # if fail, we must decrease alpha
                force_decrease = True
                # inform user
                print("Failed to compute state, adjoint and costfunctional for current potential u_next." +
                      "\n Force decrasing alpha.")
            if alpha_current < alpha_tol:
                break
            elif force_decrease:
                # decrease alpha
                alpha_current *= rho
            elif fu_bar >= fu_current + c * L2norm(g_current * (u_bar - u_current), CP.CostFunc.tvec):
                # decrase alpha
                alpha_current *= rho
            else:
                # we found alpha
                break
        iteration += 1
        # compute next step
        u_next = CP.BoxCon.projection(u_current + alpha_current * v_current)
        y_next = CP.SEq.RK4(u_next)
        p_next = CP.AEq.RK4(u_next, y_next)
        fu_next = CP.CostFunc.eval_reduced(y_next, u_next, Nq)
        # add to list
        u_list.append(u_next)
        fu_list.append(fu_next)
        print("abs(f(u_current) - f(u_next)) =", np.abs(fu_current - fu_next))
        print("L2norm(u_next - u_current) =", L2norm(u_next - u_current, CP.CostFunc.tvec))
        # check for alpha and convergence
        if alpha_current < alpha_tol:
            print("Alpha is smaller than tol, alpha =", alpha_current, ", tol = ", alpha_tol)
            break
        if L2norm(u_next - u_current, CP.CostFunc.tvec) < tol:
            # we have convergence
            print("Found u, L2norm(u_next - u_current) < tol, tol=", tol, "\n")
            break
        else:
            # setup for next step
            u_current = u_next
            y_current = y_next
            p_current = p_next
            fu_current = fu_next
    # return
    u = u_next
    y = y_next
    p = p_next
    u_list = np.asarray(u_list)
    fu_list = np.asarray(fu_list)
    return u, y, p, iteration, u_list, fu_list
