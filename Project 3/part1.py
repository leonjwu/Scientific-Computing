"""Scientific Computation Project 3, part 1
01190736
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import welch
from scipy.special import hankel1
import time


def hfield(r,th,h,levels=50,xlabel='',ylabel=''):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    return None


def repair1(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
        S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    # np.random.seed(1)
    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:
                        Bfac += B[n,j]**2
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m < p:
                    Afac = 0.0
                    Bsum = 0
                    for i in nlist[n]:
                        Afac += A[i,m]**2
                        Rsum = 0
                        for k in range(p):
                            if k != m: Rsum += A[i,k]*B[k,n]
                        Bsum += (R[i,n] - Rsum)*A[i,m]

                    B[m,n] = Bsum/(Afac+l) #New B[m,n]

        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])

    return A,B


def repair2(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    a,b = R.shape
    iK,jK = np.where(R != -1000) #indices for valid data

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))
    # R_mean = np.mean(R)
    # A[:,:] = R_mean
    # B[:,:] = R_mean

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    # np.random.seed(1)
    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: # Update A[m,n]
                    Bfac = np.sum(B[n, mlist[m]]**2)
                    Rsum = np.dot(A[m, :], B[:, mlist[m]]) - A[m, n]*B[n, mlist[m]]
                    Asum = np.dot(R[m, mlist[m]] - Rsum, B[n, mlist[m]])
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m < p: # Update B[m,n]
                    Afac = np.sum(A[nlist[n], m]**2)
                    Rsum = np.dot(A[nlist[n], :], B[:, n]) - A[nlist[n], m]*B[m, n]
                    Bsum = np.dot(R[nlist[n], n] - Rsum, A[nlist[n], m])
                    B[m,n] = Bsum/(Afac+l) #New B[m,n]
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])

    return A,B


def outwave(r0, omega=8, m=10):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0
    """
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')

    m = np.linspace(0, m)
    f = A[0,:,:]
    h1 = np.sum(hankel1(m, omega))
    h2 = np.sum(hankel1(m, omega*r0))
    B = f * h2.real / h1.real
    return B


def analyze1():
    """
    Question 1.2ii)
    Add input/output as needed
    """
    def acf(x, length=100):
        return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
            for i in range(1, length)])

    r = np.load('r.npy')
    theta = np.load('theta.npy')
    H = np.load('data3.npy')
    t = np.linspace(0, 118, 119)
    # print(np.argmin(abs(theta-np.pi/4)))
    # print(np.argmin(abs(theta-3*np.pi/4)))
    # print(np.argmin(abs(theta-5*np.pi/4)))
    # 36, 108, 180 index data
    theta1 = 36
    theta2 = 108
    theta3 = 180

    plt.figure()
    plt.contourf(t,r,H[:,theta1,:])
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title(rf'Contours of data3, $\theta =\pi /4$')
    plt.show()

    plt.figure()
    plt.contourf(t,r,H[:,theta2,:])
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title(rf'Contours of data3, $\theta =3\pi /4$')
    plt.show()


    plt.figure()
    plt.contourf(t,r,H[:,theta3,:])
    plt.xlabel('t')
    plt.ylabel('r')
    plt.title(rf'Contours of data3, $\theta =5\pi /4$')
    plt.show()

    for x in [0, len(r)//2, len(r)-1]:
        fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
        ax1.plot(t, H[x,theta1,:], label=r'$\theta =\pi /4$')
        ax1.plot(t, H[x,theta2,:], label=r'$\theta =3\pi /4$')
        ax1.plot(t, H[x,theta3,:], label=r'$\theta =5\pi /4$')
        ax1.set_xlabel('t')
        ax1.set_ylabel('h')
        ax1.set_title(f'r={r[x]}')
        ax1.legend()
        ax1.grid()
        plt.show()

        taus = 100
        fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
        ax1.plot(range(taus), acf(H[x,theta1,:]), label=r'$\theta =\pi /4$')
        ax1.plot(range(taus), acf(H[x,theta2,:]), label=r'$\theta =3\pi /4$')
        ax1.plot(range(taus), acf(H[x,theta3,:]), label=r'$\theta =5\pi /4$')
        ax1.set_xlabel('lag')
        ax1.set_ylabel(f'autocorrelation')
        ax1.set_title(f'r={r[x]}')
        ax1.legend()
        ax1.grid()
        plt.show()

        plt.figure()
        plt.xlabel(r'Frequency')
        plt.ylabel(r'Power')
        plt.title(rf'Power Spectrum of h')
        for th in [theta1, theta2, theta3]:
            X = H[x,th,:]
            w, P = welch(X)
            fr = w*1/len(t)
            plt.plot(fr,P, label=fr'$\theta = {theta[th]}$')
        plt.legend()
        plt.show()

    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
    ax1.plot(t, np.mean(np.abs(H[:,theta1,:]), axis=0), label=r'$\theta =\pi /4$')
    ax1.plot(t, np.mean(np.abs(H[:,theta2,:]), axis=0), label=r'$\theta =3\pi /4$')
    ax1.plot(t, np.mean(np.abs(H[:,theta3,:]), axis=0), label=r'$\theta =5\pi /4$')
    ax1.set_xlabel('t')
    ax1.set_ylabel('<abs(h)>')
    ax1.set_title(f'Average absolute heights over time')
    ax1.legend()
    ax1.grid()
    plt.show()

    plt.figure()
    plt.xlabel(r'Frequency')
    plt.ylabel(r'Power')
    plt.title(rf'Power Spectrum of <abs(h)> over t')
    for th in [theta1, theta2, theta3]:
        X = np.mean(np.abs(H[:,th,:]), axis=0)
        w, P = welch(X)
        fr = w*1/len(t)
        plt.plot(fr,P, label=fr'$\theta = {theta[th]}$')
    plt.legend()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
    ax1.plot(t, np.var(H[:,theta1,:], axis=0), label=r'$\theta =\pi /4$')
    ax1.plot(t, np.var(H[:,theta2,:], axis=0), label=r'$\theta =3\pi /4$')
    ax1.plot(t, np.var(H[:,theta3,:], axis=0), label=r'$\theta =5\pi /4$')
    ax1.set_xlabel('t')
    ax1.set_ylabel('var(h)')
    ax1.set_title(f'Variance of heights over time')
    ax1.legend()
    ax1.grid()
    plt.show()

    for x in [0, len(t)//2, len(t)-1]:
        fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
        ax1.plot(r, H[:,theta1,x], label=r'$\theta =\pi /4$')
        ax1.plot(r, H[:,theta2,x], label=r'$\theta =3\pi /4$')
        ax1.plot(r, H[:,theta3,x], label=r'$\theta =5\pi /4$')
        ax1.set_xlabel('r')
        ax1.set_ylabel(f'h')
        ax1.set_title(f't={t[x]}')
        ax1.legend()
        ax1.grid()
        plt.show()

        taus = 100
        print(len(r))
        fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
        ax1.plot([i*4/len(r) for i in range(taus)], acf(H[:,theta1,x]), label=r'$\theta =\pi /4$')
        ax1.plot([i*4/len(r) for i in range(taus)], acf(H[:,theta2,x]), label=r'$\theta =3\pi /4$')
        ax1.plot([i*4/len(r) for i in range(taus)], acf(H[:,theta3,x]), label=r'$\theta =5\pi /4$')
        ax1.set_xlabel('lag (r)')
        ax1.set_ylabel(f'autocorrelation')
        ax1.set_title(f't={t[x]}')
        ax1.legend()
        ax1.grid()
        plt.show()

        plt.figure()
        plt.xlabel(r'Frequency')
        plt.ylabel(r'Power')
        plt.title(rf'Power Spectrum of h')
        for th in [theta1, theta2, theta3]:
            X = H[:,th,x]
            w, P = welch(X)
            fr = w*1/len(t)
            plt.plot(fr,P, label=fr'$\theta = {theta[th]}$')
        plt.legend()
        plt.show()

    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
    ax1.plot(r, np.mean(np.abs(H[:,theta1,:]), axis=1), label=r'$\theta =\pi /4$')
    ax1.plot(r, np.mean(np.abs(H[:,theta2,:]), axis=1), label=r'$\theta =3\pi /4$')
    ax1.plot(r, np.mean(np.abs(H[:,theta3,:]), axis=1), label=r'$\theta =5\pi /4$')
    ax1.set_xlabel('r')
    ax1.set_ylabel('<abs(h)>')
    ax1.set_title(f'Average absolute heights by r')
    ax1.legend()
    ax1.grid()
    plt.show()

    plt.figure()
    plt.xlabel(r'Frequency')
    plt.ylabel(r'Power')
    plt.title(rf'Power Spectrum of <abs(h)> over r')
    for th in [theta1, theta2, theta3]:
        X = np.mean(np.abs(H[:,th,:]), axis=1)
        w, P = welch(X)
        fr = w*1/len(t)
        plt.plot(fr,P, label=fr'$\theta = {theta[th]}$')
    plt.legend()
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
    ax1.plot(r, np.var(H[:,theta1,:], axis=1), label=r'$\theta =\pi /4$')
    ax1.plot(r, np.var(H[:,theta2,:], axis=1), label=r'$\theta =3\pi /4$')
    ax1.plot(r, np.var(H[:,theta3,:], axis=1), label=r'$\theta =5\pi /4$')
    ax1.set_xlabel('r')
    ax1.set_title(f'Variance of heights by r')
    ax1.set_ylabel('var(h)')
    ax1.legend()
    ax1.grid()
    plt.show()


    return None #modify as needed


def plot_SVD3(H):
    M, N, T = H.shape
    fig1, ax1 = plt.subplots(figsize=(8,5), dpi=100)
    fig2, ax2 = plt.subplots(figsize=(8,5), dpi=100)
    fig3, ax3 = plt.subplots(figsize=(8,5), dpi=100)
    ax1.set_title(r'Cumulative Sum of Singular Values for slices in dimension 1')
    ax2.set_title(r'Cumulative Sum of Singular Values for slices in dimension 2')
    ax3.set_title(r'Cumulative Sum of Singular Values for slices in dimension 3')

    xm = float('inf')
    for m in range(M):
        U, S, Vh = np.linalg.svd(H[m,:,:])
        ax1.plot(range(1, len(S)+1), np.cumsum(S))
        x = np.where(np.cumsum(S)/np.cumsum(S)[-1]>0.999)[0][0]
        if x < xm:
            xm = x
    xm += 1
    ax1.axvline(x=xm, ls='--', color='black', label=f'i={xm}')

    xn = float('inf')
    for n in range(N):
        U, S, Vh = np.linalg.svd(H[:,n,:])
        ax2.plot(range(1, len(S)+1), np.cumsum(S))
        x = np.where(np.cumsum(S)/np.cumsum(S)[-1]>0.999)[0][0]
        if x < xn:
            xn = x
    xn += 1
    ax2.axvline(x=xn, ls='--', color='black', label=f'i={xn}')

    xt = float('inf')
    for t in range(T):
        U, S, Vh = np.linalg.svd(H[:,:,t])
        ax3.plot(range(1, len(S)+1), np.cumsum(S))
        x = np.where(np.cumsum(S)/np.cumsum(S)[-1]>0.999)[0][0]
        if x < xt:
            xt = x
    xt += 1
    ax3.axvline(x=xt, ls='--', color='black', label=f'i={xt}')

    ax1.set_xlabel('i')
    ax2.set_xlabel('i')
    ax3.set_xlabel('i')
    ax1.set_ylabel(r'$\sigma$')
    ax2.set_ylabel(r'$\sigma$')
    ax3.set_ylabel(r'$\sigma$')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()

    return xm, xn, xt


def reduce(H, inputs=(12, 69, 12), display=False):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        inputs: can be used to provide other input as needed
    Output:
        arrays: a tuple containing the arrays produced from H
    """
    M, N, T = H.shape

    if not display:
        p, q, r = inputs

    # Compute p if display
    if display:
        xm, xn, xt = plot_SVD3(H)
        # print(f'xm, xn, xt = {xm}, {xn}, {xt}')
        p = xm #12
    print(f'p={p}')

    # Reduce H, slicing in first dimension
    K = min(N, T)
    UM = np.zeros((M, N, K))
    SM = np.zeros((M, K))
    VhM = np.zeros((M, K, T))
    # Compute SVD for each slice
    for i in range(M):
        UM[i,:,:], SM[i,:], VhM[i,:,:] = np.linalg.svd(H[i,:,:], full_matrices=False)
    # Keep only relevant portions of arrays
    UM = UM[:,:,:p]
    SM = SM[:,:p]
    VhM = VhM[:,:p,:]

    # Reduce UM, slicing in p dimension
    if display:
        x1, x2, x3 = plot_SVD3(UM)
        # print(f'x1, x2, x3 = {x1}, {x2}, {x3}')
        q = x3 #69
    print(f'q={q}')
    K = min(M, N)
    UMU = np.zeros((p, M, K))
    UMS = np.zeros((p, K))
    UMVh = np.zeros((p, K, N))
    # Compute SVD for each slice
    print(UM.shape)
    for i in range(p):
        UMU[i,:,:], UMS[i,:], UMVh[i,:,:] = np.linalg.svd(UM[:,:,i], full_matrices=False)
    # Keep only relevant portions of arrays
    UMU = UMU[:,:,:q]
    UMS = UMS[:,:q]
    UMVh = UMVh[:,:q,:]

    # Reduce VhM, slicing in p dimension
    if display:
        x1, x2, x3 = plot_SVD3(VhM)
        # print(f'x1, x2, x3 = {x1}, {x2}, {x3}')
        r = x2 #12
    print(f'r={r}')
    K = min(M, T)
    VhMU = np.zeros((p, M, K))
    VhMS = np.zeros((p, K))
    VhMVh = np.zeros((p, K, T))
    # Compute SVD for each slice
    for i in range(p):
        VhMU[i,:,:], VhMS[i,:], VhMVh[i,:,:] = np.linalg.svd(VhM[:,i,:], full_matrices=False)
    # Keep only relevant portions of arrays
    VhMU = VhMU[:,:,:r]
    VhMS = VhMS[:,:r]
    VhMVh = VhMVh[:,:r,:]

    if display:
        print("Original Shape:")
        print(H.shape)
        a = H.size
        print(a)
        print("New Shapes:")
        print(UMU.shape, UMS.shape, UMVh.shape, VhMU.shape, VhMS.shape, VhMVh.shape, SM.shape)
        b = np.sum((UMU.size, UMS.size, UMVh.size, VhMU.size, VhMS.size, VhMVh.size, SM.size))
        print(b)
        print("Size of output compared to input:")
        print(f'{round(100*b/a, 3)}%')

    arrays = (UMU, UMS, UMVh, VhMU, VhMS, VhMVh, SM)
    return arrays


def reconstruct(arrays,inputs=()):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """
    # Get arrays from reduce function
    UMU, UMS, UMVh, VhMU, VhMS, VhMVh, SM = arrays
    p, M, q = UMU.shape
    N = UMVh.shape[2]
    T = VhMVh.shape[2]

    # Construct UM
    UM = np.zeros((M, N, p))
    for i in range(p):
        UM[:,:,i] = np.matmul(UMU[i,:,:], np.matmul(np.diag(UMS[i,:]), UMVh[i,:,:]))

    # Construct Vh
    VhM = np.zeros((M, p, T))
    for i in range(p):
        VhM[:,i,:] = np.matmul(VhMU[i,:,:], np.matmul(np.diag(VhMS[i,:]), VhMVh[i,:,:]))

    # Construct H
    Hnew = np.zeros((M,N,T))
    for i in range(M):
        Hnew[i,:,:] = np.matmul(UM[i,:,:], np.matmul(np.diag(SM[i,:]), VhM[i,:,:]))

    return Hnew


def plot_field(R):
    # r = np.linspace(1, 5, R.shape[0])
    # theta = np.linspace(0, 2*np.pi, R.shape[1])
    r = np.load('r.npy')
    theta = np.load('theta.npy')
    hfield(r, theta, R)


def main():

    #########
    ## 1.1 ##
    #########
    # Load and plot broken data
    R = np.load("data1.npy")
    R2 = R.copy()
    R2[R2==-1000]=np.nan
    plot_field(R2)

    # Optimal params
    t1 = time.time()
    A, B = repair2(R.copy(), p=10, l=2, niter=100)
    t2 = time.time()
    print(t2-t1)
    R2 = np.matmul(A, B)
    plot_field(R2)
    plt.show()

    # Grid search
    # p_list = [2, 4, 6, 10, 15]
    # Lambda_list = [0, 1, 2, 5]
    # niter_list = [10, 100]
    # for p in p_list:
    #     for Lambda in Lambda_list:
    #         for niter in niter_list:
    #             print(f'p={p}, lambda={Lambda}, niter={niter}')
    #
    #             t1 = time.time()
    #             A, B = repair2(R.copy(), p=p, l=Lambda, niter=niter)
    #             t2 = time.time()
    #             print(f"Fast repair: {round(t2-t1, 2)}")
    #             R2 = np.matmul(A, B)
    #             plot_field(R2)
    #             plt.title(f'p={p}, lambda={Lambda}, niter={niter}')
    #             plt.show()
    #             # plt.savefig(f'{p}{Lambda}{niter}.png')


    #########
    ## 1.2 ##
    #########
    a = outwave(1, omega=8, m=100)
    plt.plot(range(len(a)), a)
    plt.show()
    analyze1()


    #########
    ## 1.3 ##
    #########
    H = np.load("data3.npy")

    # Fast (manual inputs)
    plot_field(H[:,:,0])
    plt.show()
    H_red_arrs = reduce(H, inputs=(12, 69, 12), display=False)
    H_red = reconstruct(H_red_arrs)
    plot_field(H_red[:,:,0])
    plt.show()

    # Automatic inputs
    plot_field(H[:,:,0])
    H_red_arrs = reduce(H, display=True)
    H_red = reconstruct(H_red_arrs)
    plot_field(H_red[:,:,0])


if __name__=='__main__':
    main()
