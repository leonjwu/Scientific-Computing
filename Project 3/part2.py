"""Scientific Computation Project 3, part 2
01190736
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.spatial.distance import pdist
from scipy.signal import welch
import time


def microbes(phi,kappa,mu,L=1024,Nx=1024,Nt=1201,T=600,display=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    print("running simulation...")
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
    print("finished simulation")
    if display:
        plt.figure()
        plt.contour(x,t,f)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')

    return f,g


def newdiff(f,h):
    """
    Question 2.1 i)
    Input:
        f: array whose 2nd derivative will be computed
        h: grid spacing
    Output:
        d2f: second derivative of f computed with compact fd scheme
    """

    N = len(f)

    # Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140

    d = np.array([145/12, -76/3, 29/2, -4/3, 1/12])
    e = d[::-1]
    g = np.array([c/9, b/4, a, -2*c/9-2*b/4-2*a, a, b/4, c/9])

    # Construct banded matrix ab
    ab = np.ones((3, N))
    ab[0, 0] = 0
    ab[0, 1] = 10
    ab[0, 2:] = alpha
    ab[2, :-2] = alpha
    ab[2, -2] = 10
    ab[2, -1] = 0

    # Construct RHS b
    b = np.zeros(N)
    b[0] = np.sum(d*f[:5])
    b[1] = np.sum(g[2:]*f[:5]) + np.sum(g[:2]*f[-3:-1])
    b[2] = np.sum(g[1:]*f[:6]) + g[0]*f[-2]
    b[3:-3] = g[0]*f[:-6] + g[1]*f[1:-5] + g[2]*f[2:-4] \
        + g[3]*f[3:-3] + g[4]*f[4:-2] + g[5]*f[5:-1] + g[6]*f[6:]
    b[-3] = np.sum(g[:-1]*f[-6:]) + g[-1]*f[1]
    b[-2] = np.sum(g[:-2]*f[-5:]) + np.sum(g[-2:]*f[1:3])
    b[-1] = np.sum(e*f[-5:])

    b /= h**2

    # Enable options to enhance performance
    d2f = solve_banded((1, 1), ab, b, overwrite_ab=True, overwrite_b=True, check_finite=False)
    return d2f


def diff(f,h):
    d2f = np.zeros_like(f)
    d2f[0] = f[-2] - 2*f[0] + f[1]
    d2f[1:-1] = f[:-2] -2*f[1:-1] + f[2:]
    d2f[-1] = d2f[0]
    d2f /= h**2
    return d2f


def analyzefd():
    """
    Question 2.1 ii)
    Add input/output as needed
    """
    Nx_list = np.logspace(2, 4, 20, dtype=int) #np.logspace(1, 7, 15, dtype=int)
    h_list = 2*np.pi/(Nx_list-1)

    # Error vs x, average error vs Nx, time vs Nx
    errors_old = []
    errors_new = []
    times_old = []
    times_new = []

    for i, Nx in enumerate(Nx_list):
        # sin waves
        h = h_list[i]
        print(h, Nx)
        x = np.linspace(0, 2*np.pi, Nx)
        f = np.sin(x)
        d2f = -np.sin(x) # analytic solution

        # Old d2f
        t1 = time.time()
        d2f_old = diff(f, h)
        t2 = time.time()
        time_old = t2-t1

        # New d2f
        t1 = time.time()
        d2f_new = newdiff(f, h)
        t2 = time.time()
        time_new = t2-t1

        err_old = np.abs(d2f_old-d2f)
        err_new = np.abs(d2f_new-d2f)
        err_old_av = np.mean(err_old)
        err_new_av = np.mean(err_new)

        errors_old.append(err_old_av)
        errors_new.append(err_new_av)
        times_old.append(time_old)
        times_new.append(time_new)

        if i%5 == 0 and Nx < 1e6:
            fig0, ax0 = plt.subplots(figsize=(8,5), dpi=100)
            ax0.plot(x, d2f_old, label='Old Diff')
            ax0.plot(x, d2f_new, label='New Diff')
            ax0.plot(x, d2f, label='Analytic')
            ax0.set_title(f'f, Nx={Nx}')
            ax0.set_xlabel('x')
            ax0.set_ylabel('f')
            ax0.grid()
            ax0.legend()

            fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
            ax1.loglog(x, err_old, label='Old Diff')
            ax1.loglog(x, err_new, label='New Diff')
            ax1.set_title(f'Absolute error, Nx={Nx}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('Absolute Error')
            ax1.grid()
            ax1.legend()
            plt.show()

    errors_old = []
    errors_new = []
    times_old = []
    times_new = []
    for i, Nx in enumerate(Nx_list):
        # sin waves
        h = h_list[i]
        print(h, Nx)
        x = np.linspace(0, 2*np.pi, Nx)
        f = np.sin(100*x)/10000
        d2f = -np.sin(100*x) # analytic solution

        # Old d2f
        t1 = time.time()
        d2f_old = diff(f, h)
        t2 = time.time()
        time_old = t2-t1

        # New d2f
        t1 = time.time()
        d2f_new = newdiff(f, h)
        t2 = time.time()
        time_new = t2-t1

        err_old = np.abs(d2f_old-d2f)
        err_new = np.abs(d2f_new-d2f)
        err_old_av = np.mean(err_old)
        err_new_av = np.mean(err_new)

        errors_old.append(err_old_av)
        errors_new.append(err_new_av)
        times_old.append(time_old)
        times_new.append(time_new)

        if i%5 == 0 and Nx < 1e6:
            fig0, ax0 = plt.subplots(figsize=(8,5), dpi=100)
            ax0.plot(x, d2f_old, label='Old Diff')
            ax0.plot(x, d2f_new, label='New Diff')
            ax0.plot(x, d2f, label='Analytic')
            ax0.set_title(f'f, Nx={Nx}')
            ax0.set_xlabel('x')
            ax0.set_ylabel('f')
            ax0.grid()
            ax0.legend()

            fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
            ax1.loglog(x, err_old, label='Old Diff')
            ax1.loglog(x, err_new, label='New Diff')
            ax1.set_title(f'Absolute error, Nx={Nx}')
            ax1.set_xlabel('x')
            ax1.set_ylabel('Absolute Error')
            ax1.grid()
            ax1.legend()
            plt.show()


    P1 = np.polyfit(np.log(Nx_list), np.log(errors_old), 1)
    P2 = np.polyfit(np.log(Nx_list), np.log(errors_new), 1)
    fig2, ax2 = plt.subplots(figsize=(8,5), dpi=100)
    ax2.loglog(Nx_list, np.exp(P1[0]*np.log(Nx_list) + P1[1]), '--', label=f'grad={P1[0]}')
    ax2.loglog(Nx_list, np.exp(P2[0]*np.log(Nx_list) + P2[1]), '--', label=f'grad={P2[0]}')
    ax2.loglog(Nx_list, errors_old, label='Old Diff')
    ax2.loglog(Nx_list, errors_new, label='New Diff')
    ax2.set_title(f'Average absolute error')
    ax2.set_xlabel('Nx')
    ax2.set_ylabel('Absolute Error')
    ax2.grid()
    ax2.legend()


    P = np.polyfit(np.log(Nx_list)[8:], np.log(times_new[8:]), 1)
    fig3, ax3 = plt.subplots(figsize=(8,5), dpi=100)
    ax3.loglog(Nx_list, times_old, label='Old Diff')
    ax3.loglog(Nx_list, times_new, label='New Diff')
    ax3.loglog(Nx_list[8:], np.exp(P[0]*np.log(Nx_list[8:]) + P[1]), '--', label=f'grad={P[0]}')
    ax3.set_title(f'Timings')
    ax3.set_xlabel('Nx')
    ax3.set_ylabel('Time (s)')
    ax3.grid()
    ax3.legend()

    fig4, ax4 = plt.subplots(figsize=(8,5), dpi=100)
    ax4.plot(Nx_list, np.array(times_new)/np.array(times_old), label='New Diff Time / Old Diff Time')
    ax4.set_title(f'Timing ratio')
    ax4.set_xlabel('Nx')
    ax4.set_ylabel('Time (s)')
    ax4.grid()
    ax4.legend()
    plt.show()

    return None #modify as needed


def dynamics():
    """
    Question 2.2
    Add input/output as needed
    """

    def acf(x, length=100):
        return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
            for i in range(1, length)])

    phi = 0.3
    L = 1024
    Nx = 1024
    Nt = 1024
    kappas = [1.5,1.7,2]
    T = 600
    T_list = np.linspace(200, 6000, 3, dtype=int)
    x = np.linspace(0,L,Nx)

    for kappa in kappas:
        mu = kappa*0.4
        for T in T_list:
            Nt = T+1
            transient=100
            Nt0 = Nt-transient
            t = np.linspace(0,T,Nt)
            f, g = microbes(phi,kappa,mu,L=L,Nx=Nx,Nt=Nt,T=T,display=False)
            # Discard transient
            f, g = f[transient:,:], g[transient:,:]
            t = np.linspace(0,T,Nt-transient)

            plt.figure()
            plt.contour(x,t,f)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.title(f'Contours of f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.contour(x,t,g)
            plt.xlabel('x')
            plt.ylabel('t')
            plt.title(f'Contours of g, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.plot(x,f[-1,:])
            plt.xlabel('x')
            plt.ylabel('f')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.plot(x,np.mean(f,axis=0))
            plt.xlabel('x')
            plt.ylabel('Mean(f)')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.plot(x,np.var(f,axis=0))
            plt.xlabel('x')
            plt.ylabel('Var(f)')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.plot(t,f[:,0])
            plt.plot(t,f[:,Nx//4])
            plt.plot(t,f[:,Nx//2])
            plt.plot(t,f[:,3*Nx//4])
            plt.plot(t,f[:,Nx-1])
            plt.xlabel('t')
            plt.ylabel('f')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            plt.figure()
            plt.plot(t,np.mean(f,axis=1))
            plt.xlabel('t')
            plt.ylabel('<f>')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            taus = 100
            plt.figure()
            plt.plot(range(taus),acf(f[:,Nx//4], taus))
            plt.plot(range(taus),acf(f[:,Nx//2], taus))
            plt.plot(range(taus),acf(f[:,3*Nx//4], taus))
            plt.plot(range(taus),acf(f[:,Nx-1], taus))
            plt.xlabel('lag')
            plt.ylabel('autocorrelation')
            plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
            plt.show()

            # Fractal dimension
            X1 = f[:,  0]
            X2 =f[:, Nx//2]
            X3 = f[:,-1]
            XM = np.mean(f, axis=1)
            XS = [X1, X2, X3, XM]
            for X in XS:
                w, P = welch(X)
                fr = w*Nt/T
                plt.figure()
                plt.plot(fr,P)
                plt.xlabel(r'Frequency')
                plt.ylabel(r'Power')
                plt.title(f'Power Spectrum of X[0], kappa={kappa}, T={T}, Nt={Nt}')
                plt.legend()
                plt.show()

                f = w[P==P.max()]*Nt/T
                dt = t[1]-t[0]
                print("dt,1/f=",t[1]-t[0],1/f)
                tau = 1/(5*f)
                Del = int(tau/dt)
                print(Del,X.size)
                v1 = np.vstack([X[:-2*Del],X[Del:-Del],X[2*Del:]])

                #add code here
                plt.figure()
                plt.plot(v1[0],v1[1])
                plt.xlabel('x(t)')
                plt.ylabel(r'$x(t-\tau)$')
                plt.show()

                A = v1.T
                D = pdist(A)
                eps = np.logspace(5, -8, 50)
                C = np.zeros_like(eps)
                for i in range(len(eps)):
                    D = D[D<eps[i]]
                    C[i] = D.size
                inds = np.where((C>0) & (C<max(C)))[0]
                print(inds)
                P = np.polyfit(np.log(eps[inds]), np.log(C[inds]), 1)
                plt.figure()
                plt.loglog(eps,C)
                plt.loglog(eps[inds], np.exp(P[0]*np.log(eps) + P[1])[inds], label=f'{P[0]}')
                plt.xlabel(r'$\epsilon$')
                plt.ylabel(r'$C(\epsilon)$')
                plt.title(f'f, kappa={kappa}, T={T}, Nt={Nt}')
                plt.legend()
                plt.show()


    r = np.load('r.npy')
    theta = np.load('theta.npy')
    H = np.load('data3.npy')
    t = np.linspace(0, 119, 119)
    XS = [H]
    for i in [0, 10, 20]:
        X = H[i,i,:]

        w, P = welch(X)
        fr = w*1/119
        plt.figure()
        plt.plot(fr,P)
        plt.xlabel(r'Frequency')
        plt.ylabel(r'Power')
        plt.title(rf'Power Spectrum of data3, $\theta ={theta[i]}$, $r={r[i]}$')
        plt.legend()
        plt.show()

        f = w[P==P.max()]*Nt/T
        dt = t[1]-t[0]
        print("dt,1/f=",t[1]-t[0],1/f)
        tau = 1/(5*f)
        Del = int(tau/dt)
        print(Del,X.size)
        v1 = np.vstack([X[:-2*Del],X[Del:-Del],X[2*Del:]])

        plt.figure()
        plt.plot(v1[0],v1[1])
        plt.xlabel('x(t)')
        plt.ylabel(r'$x(t-\tau)$')
        plt.show()

        A = v1.T
        D = pdist(A)
        eps = np.logspace(5, -8, 50)
        C = np.zeros_like(eps)
        for j in range(len(eps)):
            D = D[D<eps[j]]
            C[j] = D.size
        inds = np.where((C>0) & (C<max(C)))[0]
        print(inds)
        P = np.polyfit(np.log(eps[inds]), np.log(C[inds]), 1)
        plt.figure()
        plt.loglog(eps,C)
        plt.loglog(eps[inds], np.exp(P[0]*np.log(eps) + P[1])[inds], label=f'{P[0]}')
        plt.xlabel(r'$\epsilon$')
        plt.ylabel(r'$C(\epsilon)$')
        plt.title(rf'$\theta={theta[i]}$, $r={r[i]}$')
        plt.legend()
        plt.show()

    return None #modify as needed


if __name__=='__main__':
    analyzefd()
    dynamics()
