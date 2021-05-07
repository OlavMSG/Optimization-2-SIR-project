# -*- coding: utf-8 -*-
"""
Created on 30.03.2021

@author: Olav Milian
"""
from SIR_Optimal_Control_classes import ControlProblem
from SIR_optimal_control_gradient_method import ProjectedGradientMethod
from plottingfunctions import plotAdjoint, plotRel, plotSIRu

"""
Parameters Control Problem
----------
N : int
    the total number of people.
S0 : int
    initial number of susceptible people.
I0 : int
    initial number of infected people.
R0 : int
    initial number of recovered people.
A1 : float
    coefficient in front of S in costfunctional.
A2 : float
    coefficient in front of I in costfunctional.
tau : float
    coefficient in front of u in cost functional.
v : float
    birt and death rate.
gamma : float
    recovery rate.
beta : float
    transmission coeffient.
endtime : int/float
    the endtime.
M : int
    Number of time-intervals to split [starttime, endtime] in to.
    h = (endtime - starttime) / M
starttime : int/float, optional
    the initial time. The default is 0
ul : int/float/callable, optional
    lower bound. The default is 0.0.
    if int/float make callable, if callable use it. 
up : int/float/callable, optional
    upper bound. The default is 0.9.
    if int/float make callable, if callable use it. 
use_alt_costfunc : bool, optional
    use the alternative costfunctional. The default is False.

Parameters Projected Gradient Method
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
    
Parameters plotting functions
----------
tvec : numpy.ndarray
    timestamp vector, used for the L2-norm.
y : numpy.ndarray
    3d - vector of timestamps for the state variables S, I and R.
u : numpy.ndarray
    vector of timestamps for the vaccination strategy.
p : numpy.ndarray
    3d - vector of timestamps for the adjoint variables p1, p2 and p3.
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
"""
# save the plot ???
# Note: the plot folder must exist!
save = False
# as pdf or png ???
pdf = True

"""Main program"""
def printinfo():
    """
    Function to print program info to user

    Returns
    -------
    None.

    """
    print("0: End Program")
    print("1: Run the program with the original cost functional")
    print("2: Run the program with the original cost functional")

def get_choice():
    """
    Function to get user to give a choice of what to do

    Returns
    -------
    choice : int
        The users choice is (0-2).

    """
    # print info
    printinfo()
    # posible choices
    choices = (0, 1, 2)
    # a not valid choice
    choice = -1
    # get choice from user, as long as it is invalid ask again
    while choice not in choices:
        choice = int(input("Do (0-2): "))
        if choice not in choices:
            print(choice, "is not a valid choice, must be (0-2).")
    return choice



def main():
    """
    The main function

    Returns
    -------
    None.

    """

    """General input"""
    # final Time
    T = 100
    # Number of elements
    M = 10000
    # initial susceptible, infected and recovered
    S0, I0, R0 = 700, 165, 90
    # Number of people
    N = S0 + I0 + R0
    # birt and death rate, recover rate, and transmission coefficient
    v, gamma, beta = 0.41, 0.56, 0.002
    # the used tolerance for norm(u_{k+1}-u_k)_{L2} < tol
    tol = 1e-6

    # a invalid choice
    choice = -1
    # as long as choice is not 0, meaning end loop
    while choice != 0:
        choice = get_choice()

        if choice == 1:

            print("Running program with the original costfunctional")
            # coefficient  A_1 and A_2 in cost functional
            A1, A2 = 0.050, 0.075
            # coefficient  in front of u (0 <= tau <= N)
            tau = 15
            # setup
            CP1 = ControlProblem(N, S0, I0, R0, A1, A2, tau, v, gamma, beta, T, M)

            u1, y1, p1, iterations, u_list1, fu_list1 = ProjectedGradientMethod(CP1, tol=tol)

            plotSIRu(CP1.CostFunc.tvec, y1, u1, "SIR_u_CP1", save=save, pdf=pdf)
            plotAdjoint(CP1.CostFunc.tvec, p1, "Adjoint_CP1", save=save, pdf=pdf)
            plotRel(CP1.CostFunc.tvec, u_list1, fu_list1, "RelErrs_CP1", save=save, pdf=pdf)

        elif choice == 2:

            print("Running program with the alternative costfunctional")
            # coefficient  A_1 and A_2 in cost functional
            A1, A2 = 0.050 / N * 2.95, 0.075 / N * 2.95
            # coefficient in front of u (0 <= tau <= N)
            tau = 14
            # setup
            CP2 = ControlProblem(N, S0, I0, R0, A1, A2, tau, v, gamma, beta, T, M, use_alt_costfunc=True)

            u2, y2, p2,  iterations2, u_list2, fu_list2 = ProjectedGradientMethod(CP2, tol=tol)

            plotSIRu(CP2.CostFunc.tvec, y2, u2, "SIR_u_CP2", save=save, pdf=pdf)
            plotAdjoint(CP2.CostFunc.tvec, p2, "Adjoint_CP2", save=save, pdf=pdf)
            plotRel(CP2.CostFunc.tvec, u_list2, fu_list2, "RelErrs_CP2", save=save, pdf=pdf)



# run main function
if __name__ == '__main__':
    main()