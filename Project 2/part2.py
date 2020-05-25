"""Scientific Computation Project 2, part 2
Your CID here: 01190736
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy import sparse
import matplotlib.pyplot as plt


def rwgraph(G,i0=0,M=100,Nt=100):
    """ Question 2.1
    Simulate M Nt-step random walks on input graph, G, with all
    walkers starting at node i0
    Input:
        G: An undirected, unweighted NetworkX graph
        i0: intial node for all walks
        M: Number of walks
        Nt: Number of steps per walk
    Output: X: M x Nt+1 array containing the simulated trajectories
    """
    X = np.zeros((M,Nt+1), dtype=int)

    # Initialize walks to first node
    X[:, 0] = i0
    N = G.number_of_nodes()

    # VECTORIZED
    # Generate all random numbers (flipped shape to benefit from row-major order)
    R = np.random.random((Nt, M))

    # Compute degrees
    Q = np.array([deg for (node, deg) in G.degree()])
    q_max = np.max(Q)

    # Compute adjacency list padded with zeros
    A = np.zeros((N, q_max))
    for i in range(N):
        A[i, :Q[i]] = np.array([x for x in G.adj[i]])

    for i in range(Nt):
        # Compute indices of neighbours to walk to
        S = Q[X[:, i]] * R[i, :]
        S = S.astype(int) # rounds down

        # Index adjacency list with previous time step and neighbour indices
        X[:, i+1] = A[X[:, i], S]

    # # NOT VECTORIZED
    # for j in range(M):
    #     for i in range(Nt):
    #         X[j, i+1] = np.random.choice(G.adj[X[j, i]])


    return X


def rwgraph_analyze1(input=(None)):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    # Generate Graph and auxiliary variables
    r = 20
    Nts = np.logspace(1, 4, r, dtype=int)
    Ms = np.logspace(1, 4, r, dtype=int)
    Nt_list = [500, 10000]
    M_list = [500, 10000]
    N = 2000
    L = 4
    G = nx.barabasi_albert_graph(N, L, seed=4)
    Q = G.degree()
    q_max = max([v for (k,v) in Q])
    i0 = min([k for (k,v) in Q if v == q_max])
    degree_range = np.array(range(q_max+1))
    degree_count = np.array([np.nan for i in degree_range])

    # Count degrees in graph
    for i in range(N):
        if np.isnan(degree_count[Q[i]]):
            degree_count[Q[i]] = 1
        else:
            degree_count[Q[i]] += 1

    # Calculate theoretical distribution
    degree_theo = degree_range / G.number_of_edges() / 2


    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(111)
    X = rwgraph(G,i0=i0,M=10000,Nt=10000)

    for Nt in Nt_list:
        errors = []
        for M in Ms:
            print(Nt, M)
            walk_degree_count = np.array([0. for i in degree_range])

            # Count degrees of ending nodes of walk
            for i in range(M):
                walk_degree_count[Q[X[i, Nt]]] += 1
            for i in degree_range:
                if walk_degree_count[i] == 0:
                    walk_degree_count[i] = np.nan

            walk_degree_count_norm = walk_degree_count / degree_count / M
            # Compute error
            error = 0
            for i in degree_range:
                y = walk_degree_count_norm[i]
                if not np.isnan(y):
                    error += (y - degree_theo[i])**2 * walk_degree_count[i]
            errors.append(error)

            if (M == 10000 and Nt == 10000) or (M == 545 and Nt == 500):
                ax1.plot(degree_range, walk_degree_count_norm, 'x', label=f"Nt={Nt}, M={M}")

        ax2.loglog(Ms, errors, marker='x', linestyle='dashed', label=f'Nt={Nt}')


    for M in M_list:
        errors = []
        for Nt in Nts:
            print(Nt, M)
            walk_degree_count = np.array([np.nan for i in degree_range])

            # Count degrees of ending nodes of walk
            for i in range(M):
                if np.isnan(walk_degree_count[Q[X[i, Nt]]]):
                    walk_degree_count[Q[X[i, Nt]]] = 1
                else:
                    walk_degree_count[Q[X[i, Nt]]] += 1

            walk_degree_count_norm = walk_degree_count / degree_count / M
            # Compute error
            error = 0
            for i in degree_range:
                y = walk_degree_count_norm[i]
                if not np.isnan(y):
                    error += (y - degree_theo[i])**2 * walk_degree_count[i]
            errors.append(error)

        ax2.loglog(Nts, errors, marker='x', linestyle='-', label=f'M={M}')


    ax1.plot(degree_range, degree_theo, '--', label='Asymptotic Result')
    ax1.legend()
    ax1.set_title(f'Proportion of Random Walk End Positions for Node Degrees')
    ax1.set_xlabel('Final Node Degree of Walker')
    ax1.set_ylabel('Proportion of Walkers')
    ax1.grid()

    ax2.legend()
    ax2.set_title(f'Difference to Asymptotic Result for increasing M (dashed lines) and Nt (filled lines)')
    ax2.set_xlabel('M=Nt')
    ax2.set_ylabel('Difference to Asymptotic Result')
    ax2.grid()

    plt.show()

    return None


def rwgraph_analyze2(input=(None)):
    """Analyze similarities and differences
    between simulated random walks and linear diffusion on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    # Generate graph
    M = 10000
    Nt = 10000
    N = 2000
    L1 = 4
    G = nx.barabasi_albert_graph(N, L1, seed=4)
    A = nx.adjacency_matrix(G)
    Q0 = G.degree()
    q_max = max([v for (k,v) in Q0])
    i0 = min([k for (k,v) in Q0 if v == q_max])
    degree_range = np.array(range(q_max+1))
    degree_count = np.array([np.nan for i in degree_range])

    # Count degrees in graph
    for i in range(N):
        if np.isnan(degree_count[Q0[i]]):
            degree_count[Q0[i]] = 1
        else:
            degree_count[Q0[i]] += 1

    X = rwgraph(G,i0=i0,M=M,Nt=Nt)

    # Theoretical
    degree_theo = degree_range / G.number_of_edges() / 2

    # Calculate degrees
    q = np.array([deg for (node, deg) in G.degree()])
    Q = sparse.diags([q], [0])
    Q_inv = sparse.diags([1/q], [0])
    ones = sparse.diags([np.ones(N)], [0])

    # Calculate linear operators
    L = nx.laplacian_matrix(G)
    Ls = ones - Q_inv*A
    Lst = sparse.csr_matrix.transpose(Ls)


    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twinx()
    n = N

    # eL, vL = sparse.linalg.eigs(-L, k=n)
    # eLs, vLs = sparse.linalg.eigs(-Ls, k=n)
    # eLst, vLst = sparse.linalg.eigs(-Lst, k=n)

    # Compute eigen vectors and values
    eL, vL = np.linalg.eig(-L.toarray())
    eLs, vLs = np.linalg.eig(-Ls.toarray())
    eLst, vLst = np.linalg.eig(-Lst.toarray())

    # Sort eigenvalues
    eL_sorted = sorted(eL, reverse=True)
    eLs_sorted = sorted(eLs, reverse=True)
    eLst_sorted = sorted(eLst, reverse=True)

    # Plot eigenvalues
    ax1.plot(range(n), eL_sorted, 'x', ms=1.5, label=r'$L$')
    ax2.plot(range(n), eLs_sorted, 'x', ms=1.5, color='orange', label=r'$L_s$ and $(L_s)^T $')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_title(f'Eigenvalues for Linear Spreading Operators')
    ax1.set_xlabel('')
    ax1.set_ylabel(r'Eigenvalues ($L$)')
    ax2.set_ylabel(r'Eigenvalues ($L_s$ and $(L_s)^T $')
    ax1.grid()
    plt.show()

    # Set up solutions
    f0 = np.zeros(N)
    f0[i0] = 1
    t = np.array(range(Nt+1))

    def RHS(f, t, L):
        return -L.dot(f)

    # Solutions to ODEs
    fL = odeint(RHS, f0.copy(), t, args=(L,))[-1, :]
    fLs = odeint(RHS, f0.copy(), t, args=(Ls,))[-1, :]
    fLst = odeint(RHS, f0.copy(), t, args=(Lst,))[-1, :]

    # Count degrees of ending nodes of walk
    walk_degree_count = np.array([0. for i in degree_range])
    L_degree_count = walk_degree_count.copy()
    Ls_degree_count = walk_degree_count.copy()
    Lst_degree_count = walk_degree_count.copy()

    # Count degrees
    for i in range(M):
        walk_degree_count[Q0[X[i, -1]]] += 1
    for i in range(N):
        L_degree_count[Q0[i]] += fL[i]
        Ls_degree_count[Q0[i]] += fLs[i]
        Lst_degree_count[Q0[i]] += fLst[i]

    # Clean data
    walk_degree_count[walk_degree_count == 0] = np.nan
    L_degree_count[walk_degree_count == 0] = np.nan
    Ls_degree_count[walk_degree_count == 0] = np.nan
    Lst_degree_count[walk_degree_count == 0] = np.nan

    # Scale data
    walk_degree_count_norm = walk_degree_count / degree_count / M
    L_degree_count_norm = L_degree_count / degree_count
    Ls_degree_count_norm = Ls_degree_count / degree_count
    Lst_degree_count_norm = Lst_degree_count / degree_count

    fig2 = plt.figure(figsize=(10,6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(degree_range, walk_degree_count_norm, 'x', label=f"Random Walk, M={M}, Nt={Nt}")
    ax2.plot(degree_range, L_degree_count_norm, 'x', label=r'$L$')
    ax2.plot(degree_range, Ls_degree_count_norm, 'x', label=r'$L_s$')
    ax2.plot(degree_range, Lst_degree_count_norm, 'x', label=r'$(L_s)^T$')
    ax2.plot(degree_range, degree_theo, '--', label='Asymptotic Result (Random Walk)')
    ax2.legend()
    ax2.set_title(f'Proportion of End Positions for Node Degrees, Nt={Nt}, M={M}')
    ax2.set_xlabel('Final Node Degree of Walker')
    ax2.set_ylabel('Proportion of Walkers / Intensity')
    ax2.grid()
    plt.show()

    return None


def modelA(G,x=0,i0=0.1,beta=1.0,gamma=1.0,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    beta,gamma: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """

    N = G.number_of_nodes()

    # Precompute values
    tarray = np.linspace(0,tf,Nt+1)
    A = nx.adjacency_matrix(G)
    gammaA = A.multiply(gamma)
    mbeta = -beta

    def RHS(y,t):
        """Compute RHS of modelA at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt

        Discussion: add discussion here
        """
        gammaAi = gammaA.dot(y)
        return mbeta*y + np.multiply(gammaAi, (1-y))

    # Initial conditions
    y = np.zeros(N)
    y[x] = i0

    # Compute solution
    jarray = odeint(RHS, y, tarray)
    iarray = np.transpose(jarray)

    return iarray


def modelB(G,x=0,i0=0.1,alpha=-0.01,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model B

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    gamma: model parameter
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
    each time step.
    """
    # Create graph
    N = G.number_of_nodes()
    N2 = 2*N
    tarray = np.linspace(0,tf,Nt+1)

    # Precompute values
    A = nx.adjacency_matrix(G)
    q = np.array([deg for (node, deg) in G.degree()])
    Q = sparse.diags([q], [0])
    L = nx.laplacian_matrix(G)
    alphaL = L.multiply(alpha)

    def RHS(y,t):
        """Compute RHS of modelB at time t
        input: y should be a size N array
        output: dydt, a size 2N array corresponding to ds/dt, di/dt

        Discussion: add discussion here
        """
        # Note second term in sum simplifies to 0
        dydt[:N] = y[N:]
        dydt[N:] = alphaL.dot(y[:N])
        return dydt

    dydt = np.zeros(N2)
    y = np.zeros((N2))
    y[N+x] = i0

    # Compute solution
    jarray = odeint(RHS, y, tarray)
    iarray = np.transpose(jarray)
    return iarray


def transport(input=(None)):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    # Create graph
    N = 100
    M = 5
    Nt = 1000
    tf = 50
    i0 = 0.1
    alpha = -0.01
    beta = 0.5
    gamma = 0.1
    G = nx.barabasi_albert_graph(N, M, seed=2)
    L = nx.laplacian_matrix(G)
    Q0 = G.degree()
    q_max = max([v for (k,v) in Q0])
    x = min([k for (k,v) in Q0 if v == q_max])
    degree_range = np.array(range(q_max+1))
    degree_count = np.array([np.nan for i in degree_range])

    # Count degrees in graph
    for i in range(N):
        if np.isnan(degree_count[Q0[i]]):
            degree_count[Q0[i]] = 1
        else:
            degree_count[Q0[i]] += 1

    # Graph for intensity
    Q = np.array([deg for (node, deg) in Q0])
    A = modelA(G,x=x,i0=i0,beta=beta,gamma=gamma,tf=tf,Nt=Nt)
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(np.linspace(0, tf, Nt+1), np.transpose(A))
    ax1.set_title(rf'Model A, $N$={N}, $\beta$={beta}, $\gamma$={gamma}, tf={tf}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity')
    ax1.grid()
    plt.show()

    # For different params
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    for gamma in [1, 0.1, 0.01]:
        for beta in [0.1, 0.5, 1]:
            A = modelA(G,x=x,i0=i0,beta=beta,gamma=gamma,tf=tf,Nt=Nt)
            ax1.plot(np.linspace(0, tf, Nt+1), np.mean(A, axis=0), label=rf'$\beta$={beta}, $\gamma$={gamma}')
    ax1.legend()
    ax1.set_title(rf'Model A, $N$={N}, tf={tf}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Intensity')
    ax1.grid()
    plt.show()


    # Model B graph
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    tf = 500
    alpha = -0.01
    B = modelB(G,x=x,i0=i0,alpha=alpha,tf=tf,Nt=Nt)
    ax1.set_title(rf'Model B, $N$={N}, $\alpha$={alpha}, tf={tf}')
    ax1.legend()
    ax1.plot(np.linspace(0, tf, Nt+1), np.transpose(B[N:]))
    ax1.grid()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity')
    plt.show()


    # Model B amplitude
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    tf = 500
    alpha = -0.01
    B = modelB(G,x=x,i0=i0,alpha=alpha,tf=tf,Nt=Nt)
    ax1.loglog(Q, np.max(B[N:], axis=1),'x', label=f'tf={tf}')
    ax1.set_title(rf'Model B maximum amplitude vs degree, $N$={N}, $\alpha$={alpha}')
    ax1.grid()
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Maximum Amplitude')
    ax1.legend()
    plt.show()


    # Model A intensity2
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    beta = 0.5
    for gamma in [0.02, 0.05, 0.1, 0.2]:
        A = modelA(G,x=x,i0=i0,beta=beta,gamma=gamma,tf=tf,Nt=Nt)
        # Count degrees of ending nodes of walk
        A_degree_count = np.array([0. for i in degree_range])
        for i in range(N):
            A_degree_count[Q0[i]] += A[i, -1]
        A_degree_count[A_degree_count == 0] = np.nan
        A_degree_count_norm = A_degree_count / degree_count
        ax1.plot(degree_range, A_degree_count_norm, 'x', label=rf'$\gamma$={gamma}')
        if gamma == 0.05:
            idx = np.isfinite(A_degree_count_norm)
            P = np.polyfit([i for i in degree_range if idx[i]], A_degree_count_norm[idx], 1)
            plt.plot(degree_range, P[1] + [P[0]*i for i in degree_range], '--', label=rf'Linear Fit, $\gamma = ${gamma}')
    plt.title(rf'Model A Intensity vs Degree $\beta$={beta}')
    ax1.grid()
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Intensity')
    plt.legend()
    plt.show()


    # Solution for Linear Diffusion
    def RHS(f, t, L):
        return -D*L.dot(f)

    tf = 25
    alpha = -0.01
    beta = 0.5
    gamma = 0.1
    i0 = 1
    D = 0.005

    # Linear Diffusion Solutions
    f0 = np.zeros(N)
    f0[x] = i0
    t = np.linspace(0, tf, Nt+1)
    fL = odeint(RHS, f0.copy(), t, args=(L,))

    A = modelA(G,x=x,i0=i0,beta=beta,gamma=gamma,tf=tf,Nt=Nt)
    B = modelB(G,x=x,i0=i0,alpha=alpha,tf=tf,Nt=Nt)
    fig1 = plt.figure(figsize=(10,6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t, np.var(A, axis=0), label='Model A')
    ax1.plot(t, np.var(B[N:], axis=0), label='Model B')
    ax1.plot(t, np.var(fL, axis=1), label='Linear Diffusion')
    ax1.legend()
    ax1.set_title(f'Variance over time, tf={tf}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variance')
    ax1.grid()
    plt.show()

    return None


if __name__=='__main__':
    rwgraph_analyze1()
    rwgraph_analyze2()
    transport()
