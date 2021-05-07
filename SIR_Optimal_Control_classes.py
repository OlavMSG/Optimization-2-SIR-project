# -*- coding: utf-8 -*-
"""
Created on 10.03.2021

@author: Olav Milian
Used coded from Tor Ola in RK4 functions
"""
import numpy as np
from numba import njit
from scipy.special import roots_legendre
from scipy.interpolate import interp1d


class StateEq:
    """
    Class for the State Equation
    y'(t) = A(u(t)) @ y(t) + F(y(t))
    y(0) = y0
    """

    def __init__(self, N, S0, I0, R0, v, gamma, beta, endtime, M, starttime=0):
        """
        Default function to set the constants used


        Parameters
        ----------
        N : int
            the total number of people.
        S0 : int
            initial number of susceptible people.
        I0 : int
            initial number of infected people.
        R0 : int
            initial number of recovered people.
        v : float
            birt and death rate.
        gamma : float
            recovery rate.
        beta : float
            transmission coefficient.
        endtime : int/float
            the endtime.
        M : int
            Number of time-intervals to split [starttime, endtime] in to.
            h = (endtime - starttime) / M
        starttime : int/float, optional
            the initial time. The default is 0

        Returns
        -------
        None.

        """
        # the start state (# initial susceptible, infected and recovered)
        self.y0 = np.array([S0, I0, R0])
        # Number of people
        self.N = N
        # birt and death rate
        self.v = v
        self.vN = v * N
        # recover rate
        self.gamma = gamma
        # transmission coeff. beta / N
        self.betaN = beta / N
        # start and end time
        self.starttime = starttime
        self.endtime = endtime
        # Number of elements
        self.M = M
        # step size divided by 6
        self.h = (endtime - starttime) / M
        self.h6 = self.h / 6
        # time stamp vector
        # self.tvec = np.linspace(starttime, endtime, M + 1)

    @staticmethod
    @njit()
    def A(y, u_m, v, gamma):
        """
        Function implementing the effect of the matrix A(u(t)) on y

        Parameters
        ----------
        y : numpy.ndarray
            state at time t_m, dim (3,1).
        u_m : float
            vaccination strategy at time t_m.
        v : float
            birt and death rate.
        gamma : float
            recovery rate.

        Returns
        -------
        a : numpy.ndarray
            the vector A(u(t))y, dim (3,1).

        """
        a = np.zeros(3, dtype=np.float64)
        a[0] = -(v + u_m) * y[0]
        a[1] = -(gamma + v) * y[1]
        a[2] = u_m * y[0] + gamma * y[1] - v * y[2]
        return a

    @staticmethod
    @njit()
    def F(y, betaN, vN):
        """
        Function implementing the vector F(y(t))
        
        Parameters
        ----------
        y : numpy.ndarray
            state at time t_m, dim (3,1).
        betaN : float
            transmission coefficient divided by N.
            N is the total population.
        vN : float
            birth and death rate times N.
            N is the total population.

        Returns
        -------
        Fy : numpy.ndarray
            Vector F(Y(t)), dim (3,1).

        """
        
        SI = y[0] * y[1]
        Fy = betaN * np.array([-SI, SI, 0.0])
        Fy[0] += vN
        return Fy

    def f(self, y, u_m):
        """
        Function f for the state equation
        y'(t) = A(u(t)) @ y(t) + F(y(t))

        Parameters
        ----------
        y : numpy.ndarray
            state at time t_m, dim (3,1).
        u_m : float
            vaccination strategy at time t_m.

        Returns
        -------
        numpy.ndarray
            value of the function f for the state equation.

        """
        
        return self.A(y, u_m, self.v, self.gamma) + self.F(y, self.betaN, self.vN)

    def RK4(self, uh):
        """
        Forward-time Forth order Runge Kutta method for the state equation.
        By Tor Ola, rewritten to work here.

        Parameters
        ----------
        uh : numpy.ndarray
            vector of time stamps for the current vaccination strategy.

        Raises
        ------
        ValueError
            If NaN value in Y is detected.

        Returns
        -------
        Y : numpy.ndarray
            3d-array of the timestamps for the state variables.

        """

        Y = np.zeros((3, self.M + 1), dtype=float)  # State vectors
        Y[:, 0] = self.y0  # Initialize state vectors
        for m in range(self.M):
            Ym = Y[:, m]  # Get most current update of statevectors.
            if np.isnan(Ym).any():
                raise ValueError(
                    "NaN value detected; The stepsize migth too lagre, try increasing the number of elements M, to get a smaller h")
            # tm = self.tvec[m]  # get current time
            # get the vaccination protocols
            um1 = uh[m]  # Get current vaccination protocol
            um4 = uh[m + 1]  # Get next vaccination protocol
            um2 = 0.5 * (um1 + um4)  # interpolate
            # Perform RK4 update
            K1 = self.f(Ym, um1)
            K2 = self.f(Ym + 0.5 * K1, um2)
            K3 = self.f(Ym + 0.5 * K2, um2)
            K4 = self.f(Ym + K3, um4)
            Y[:, m + 1] = Ym + self.h6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return Y


class AdjointEq:
    """
    class for the Adjoint equation
    -p'(t) = A(u(t))^T @ p(t) + J_F(y(t))^T @ p(t) + b
    p(T) = 0
    """

    def __init__(self, N, A1, A2, v, gamma, beta, endtime, M, starttime=0, use_alt_costfunc=False):
        """
        Default function to set the constants used

        Parameters
        ----------
        N : int
            the total number of people.
        A1 : float
            coefficient in front of S in costfunctional.
        A2 : float
            coefficient in front of I in costfunctional.
        v : float
            birt and death rate.
        gamma : float
            recovery rate.
        beta : float
            transmission coefficient.
        endtime : int/float
            the endtime.
        M : int
            Number of time-intervals to split [starttime, endtime] in to.
            h = (endtime - starttime) / M
        starttime : int/float, optional
            the initial time. The default is 0
        use_alt_costfunc : bool, optional
            use the alternative costfunctional. The default is False.

        Returns
        -------
        None.

        """
        # default function to set the constants used

        # Number of people
        self.N = N
        # coff. A_1 and A_2 in cost functional, placed in vector b1
        self.b1 = np.array([A1, A2, 0.0])
        # birt and death rate
        self.v = v
        # recover rate
        self.gamma = gamma
        # transmission coeff. beta / N
        self.betaN = beta / N
        # the end state
        self.pT = np.zeros(3, dtype=np.float64)
        # start and end time
        self.starttime = starttime
        self.endtime = endtime
        # Number of elements
        self.M = M
        # step size divided by 6
        self.h = (endtime - starttime) / M
        self.h6 = self.h / 6
        # time stamp vector
        # self.tvec = np.linspace(starttime, endtime, M + 1)
        # set Cost-Func. to use
        if use_alt_costfunc:
            self.f = self.f2
        else:
            self.f = self.f1

    @staticmethod
    @njit()
    def AT(p, u_m, v, gamma):
        """
        Function implementing the effect of the matrix A(u(t))^T  on p(t)

        Parameters
        ----------
        p : numpy.ndarray
            adjoint state at time t_m.
        u_m : float
            vaccination strategy at time t_m.
        v : float
            birt and death rate.
        gamma : float
            recovery rate.

        Returns
        -------
        a : numpy.ndarray
            the vector A(u(t))y, dim (3,1).

        """
    
        a = np.zeros(3, dtype=np.float64)
        a[0] = -(v + u_m) * p[0] + u_m * p[2]
        a[1] = -(gamma + v) * p[1] + gamma * p[2]
        a[2] = - v * p[2]
        return a

    @staticmethod
    @njit()
    def J_FT(p, y, betaN):
        """
        Function implementing the effect of the jacobian matrix J_F(y(t))^T on p(t)

        Parameters
        ----------
        p : numpy.ndarray
            adjoint state at time t_m.
        y : numpy.ndarray
            state at time t_m.
        betaN : float
            transmission coefficient divided by N.
            N is the total population.

        Returns
        -------
         : numpy.ndarry
            the vector J_F(y(t))^Tp, dim (3,1).

        """
        a = np.array([y[1], y[0], 0.0])
        a *= betaN * (-p[0] + p[1])
        return a

    def f1(self, phi, u_m, y):
        """
        f for rewritten adjont eq. to forward time, for original cost Func
        phi'(t)= A(u(T-t)) ^ T @ phi(t) + J_F(y(T-t)) ^ T @ phi(t) + b1

        Parameters
        ----------
        phi : numpy.ndarray
            adjoint state p(t-T) for forward time, dim(3,1).
        u_m : float
            vaccination strategy at time t_m.
        y : numpy.ndarray
            state at time t_m.

        Returns
        -------
        numpy.ndarray
            value of the function f for the adjoint equation.

        """
        
        return self.AT(phi, u_m, self.v, self.gamma) + self.J_FT(phi, y, self.betaN) + self.b1

    def f2(self, phi, u_m, y):
        """
        f for rewritten adjont eq. to forward time, for alternative cost Func.
        phi'(t)= A(u(T-t)) ^ T @ phi(t) + J_F(y(T-t)) ^ T @ phi(t) + b2(y(T-t))
        b2(y(T-t)) = b1 * y(T-t)

        Parameters
        ----------
        phi : numpy.ndarray
            adjoint state p(t-T) for forward time, dim(3,1).
        u_m : float
            vaccination strategy at time t_m.
        y : numpy.ndarray
            state at time t_m.

        Returns
        -------
        numpy.ndarray
            value of the function f for the adjoint equation.

        """
        
        return self.AT(phi, u_m, self.v, self.gamma) + self.J_FT(phi, y, self.betaN) + self.b1 * y

    def RK4(self, uh, yh):
        """
        Backward-time Forth order Runge Kutta method for the adjoint equation.
        By Tor Ola, rewritten to work here.

        Parameters
        ----------
        uh : numpy.ndarray
            vector of time stamps for the current vaccination strategy.
        yh : numpy.ndarray
            3d-array of the timestamps for the state variables..

        Raises
        ------
        ValueError
            If NaN value in P is detected..

        Returns
        -------
        P : numpy.ndarray
            3d-array of the timestamps for the adjoint variables.

        """

        P = np.zeros((3, self.M + 1), dtype=np.float)  # Adjoint vectors
        # P[:, M] = self.pT  # Initialize adjoint vectors. it is Zeros, so line is not needed.
        for m in range(self.M, 0, -1):  # move backwards, here "Pm = phi"
            Pm = P[:, m]  # Get most current update of statevectors.
            if np.isnan(Pm).any():
                raise ValueError(
                    "NaN value detected; The stepsize migth too lagre, try increasing the number of elements M, to get a smaller h")
            # tm = self.tvec[m]  # get current time
            # get the vaccination protocols
            um4 = uh[m]  # Get current vaccination protocol
            um1 = uh[m - 1]  # Get previous vaccination protocol
            um2 = 0.5 * (um1 + um4)  # interpolate
            # get the states
            y4 = yh[:, m]  # Get current state
            y1 = yh[:, m - 1]  # Get previous state
            y2 = 0.5 * (y1 + y4)  # interpolate
            # Perform RK4 update
            K1 = self.f(Pm, um1, y1)
            K2 = self.f(Pm + 0.5 * K1, um2, y2)
            K3 = self.f(Pm + 0.5 * K2, um2, y2)
            K4 = self.f(Pm + K3, um4, y4)
            P[:, m - 1] = Pm + self.h6 * (K1 + 2 * K2 + 2 * K3 + K4)
        return P


class BoxConstraints:
    """
    class to hold the discrete box constraints
    """

    def __init__(self, endtime, M, ul=0.0, up=0.9, starttime=0):
        """
        default function to set the lower and upper bound functions

        Parameters
        ----------
        endtime : int/float
            the endtime.
        M : int
            Number of time-intervals to split [starttime, endtime] in to.
            h = (endtime - starttime) / M
        ul : int/float/callable, optional
            lower bound. The default is 0.0.
            if int/float make callable, if callable use it. 
        up : int/float/callable, optional
            upper bound. The default is 0.9.
            if int/float make callable, if callable use it. 
        starttime : int/float, optional
            the initial time. The default is 0

        Raises
        ------
        ValueError
            if ul or up is not int, float or callable.

        Returns
        -------
        None.

        """

        # if function set it
        # elseif int or float make the function
        # else raise error
        if callable(ul):
            self.u_lower = ul
        elif isinstance(ul, (int, float)):
            self.u_lower = lambda t: ul
        else:
            raise ValueError("ul is not a callable object, int nor float")

        if callable(up):
            self.u_upper = up
        elif isinstance(up, (int, float)):
            self.u_upper = lambda t: up
        else:
            raise ValueError("up is not a callable object, int nor float")

        tvec = np.linspace(starttime, endtime, M + 1)
        # lower bound
        self.ul_discrete = self.u_lower(tvec)
        # upper bound
        self.up_discrete = self.u_upper(tvec)

    @staticmethod
    @njit()
    def _projection(x, ul_discrete, up_discrete):
        """
        Function to run projection

        Parameters
        ----------
        x : numpy.ndarray
            vector of values to be projected onto the admissible set.
        ul_discrete : numpy.ndarray
            discrete lower bound.
        up_discrete : numpy.ndarray
            discrete upper bound.

        Returns
        -------
        x_ad: numpy.ndarray
            vector x projected onto the admissible set.

        """

        x_ad = np.fmin(up_discrete, np.fmax(x, ul_discrete))
        return x_ad

    def projection(self, x):
        """
        The projection operator

        Parameters
        ----------
        x : numpy.ndarray
            vector of values to be projected onto the admissible set.

        Returns
        -------
        numpy.ndarray
            vector x projected onto the admissible set.

        """
        return self._projection(x, self.ul_discrete, self.up_discrete)


class CostFunctional:
    def __init__(self, endtime, A1, A2, tau, M, starttime=0, use_alt_costfunc=False):
        """
        default function to set the constant used

        Parameters
        ----------
        endtime : int/float
            the endtime.
        A1 : float
            coefficient in front of S in costfunctional.
        A2 : float
            coefficient in front of I in costfunctional.
        tau : float
            coefficient in front of u in cost functional.
        M : int
            Number of time-intervals to split [starttime, endtime] in to.
            h = (endtime - starttime) / M
        starttime : int/float, optional
            the initial time. The default is 0
        use_alt_costfunc : bool, optional
            use the alternative costfunctional. The default is False.


        Returns
        -------
        None.

        """

        # start and end time
        self.starttime = starttime
        self.endtime = endtime
        # coeff A_1 and A_2 in cost functional
        self.A1 = A1
        self.A2 = A2
        # constant in front of u in cost functional
        self.tau = tau
        # time stamp vector
        self.tvec = np.linspace(starttime, endtime, M + 1)
        # use the 1. cost func
        if use_alt_costfunc:
            self.eval_reduced = self.eval_reduced2
        else:
            self.eval_reduced = self.eval_reduced1

    def eval_reduced1(self, y, uh, Nq):
        """
        Function to do a numerical evaluation of the original reduced costfunctional,
        given the discrete functions for y and u, which will be interpolated.
        The integral is done using Gaussian quadrature

        Parameters
        ----------
        y: numpy.ndarray
            2D array with rows [S, I, R], each row is a time stamp
        uh: numpy.ndarray
            1D array of time stamps of the control function
        Nq : int
            How many points to use in the numerical integration, Nq-point rule.

        Returns
        -------
        fu : float
            value of the cost function (integral).
        """

        # Weights and gaussian quadrature points, also works for Nq larger than 4
        z_q, rho_q = roots_legendre(Nq)

        # interpolate S, I and u
        S = interp1d(self.tvec, y[0, :], assume_sorted=True)
        I = interp1d(self.tvec, y[1, :], assume_sorted=True)
        u = interp1d(self.tvec, uh, assume_sorted=True)

        # the function to integrate for the cost functional
        def phi(t):
            return self.A1 * S(t) + self.A2 * I(t) + 0.5 * self.tau * u(t) ** 2

        # compute the integral numerically
        fu = (self.endtime - self.starttime) * 0.5 * \
             np.sum(rho_q * phi(0.5 * (self.endtime - self.starttime) * z_q + 0.5 * (self.endtime + self.starttime)))
        return fu

    def eval_reduced2(self, y, uh, Nq):
        """
        Function to do a numerical evaluation of the alternative reduced costfunctional,
        given the discrete functions for y and u, which will be interpolated.
        The integral is done using Gaussian quadrature

        Parameters
        ----------
        Nq : int
            How many points to use in the numerical integration, Nq-point rule.
        y: numpy.ndarray
            2D array with rows [S, I, R], each row is a time stamp
        uh: numpy.ndarray
            1D array of time stamps of the control function
        Nq : int
            How many points to use in the numerical integration, Nq-point rule.

        Returns
        -------
        fu : float
            value of the cost function (integral).
        """
        # Weights and gaussian quadrature points, also works for Nq larger than 4
        z_q, rho_q = roots_legendre(Nq)

        # interpolate S, I and u
        S = interp1d(self.tvec, y[0, :], assume_sorted=True)
        I = interp1d(self.tvec, y[1, :], assume_sorted=True)
        u = interp1d(self.tvec, uh, assume_sorted=True)

        # the function to integrate for the costfunctional
        def phi(t):
            return 0.5 * self.A1 * S(t) ** 2 + 0.5 * self.A2 * I(t) ** 2 + 0.5 * self.tau * u(t) ** 2

        # compute the integral numerically
        fu = (self.endtime - self.starttime) * 0.5 * \
             np.sum(rho_q * phi(0.5 * (self.endtime - self.starttime) * z_q + 0.5 * (self.endtime + self.starttime)))
        return fu

    def gradient(self, y, p, u):
        """
        Function to get the gradient of the reduced costfunctionals

        Parameters
        ----------
        y : numpy.ndarray
            3d-array of the timestamps for the state variables.
        p : numpy.ndarray
            3d-array of the timestamps for the adjoint variables.
        u: numpy.ndarray
            1D array of time stamps of the control function

        Returns
        -------
        numpy.ndarray
            1D array of time stamps of the gradient.

        """
        # 
        return self.tau * u + (-p[0, :] + p[2, :]) * y[0, :]


class ControlProblem:
    """
    Main class implementing a control problem, using the above classes
    """
    def __init__(self, N, S0, I0, R0, A1, A2, tau, v, gamma, beta, endtime, M,
                 starttime=0, ul=0.0, up=0.9, use_alt_costfunc=False):
        """
        Default function to set the constants in the 4 classes making a control problem

        Parameters
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

        Returns
        -------
        None.

        """
        self.SEq = StateEq(N, S0, I0, R0, v, gamma, beta, endtime, M, starttime=starttime)
        self.AEq = AdjointEq(N, A1, A2, v, gamma, beta, endtime, M, use_alt_costfunc=use_alt_costfunc)
        self.BoxCon = BoxConstraints(endtime, M, ul=0.0, up=0.9, starttime=starttime)
        self.CostFunc = CostFunctional(endtime, A1, A2, tau, M, starttime=starttime, use_alt_costfunc=use_alt_costfunc)
        if use_alt_costfunc:
            print("Using the Alternative Costfunctional")
        else:
            print("Using the Original Costfunctional")
