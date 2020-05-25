""" Your college id here: 01190736
    Template code for part 1, contains 4 functions:
    newSort, merge: codes for part 1.1
    time_newSort: to be completed for part 1.1
    findTrough: to be completed for part 1.2
"""

import time
import numpy as np
import matplotlib.pyplot as plt


def newSort(X,k=0):
    """Given an unsorted list of integers, X,
        sort list and return sorted list
    """
    n = len(X)
    if n==1:
        return X
    elif n<=k:
        for i in range(n-1):
            ind_min = i
            for j in range(i+1,n):
                if X[j]<X[ind_min]:
                    ind_min = j
            X[i],X[ind_min] = X[ind_min],X[i]
        return X
    else:
        L = newSort(X[:n//2],k)
        R = newSort(X[n//2:],k)
        return merge(L,R)


def merge(L,R):
    """Merge 2 sorted lists provided as input
    into a single sorted list
    """
    M = [] #Merged list, initially empty
    indL,indR = 0,0 #start indices
    nL,nR = len(L),len(R)

    #Add one element to M per iteration until an entire sublist
    #has been added
    for i in range(nL+nR):
        if L[indL]<R[indR]:
            M.append(L[indL])
            indL = indL + 1
            if indL>=nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR>=nR:
                M.extend(L[indL:])
                break
    return M


def time_newSort(inputs=None):
    """Analyze performance of newSort
    Use variables inputs and outputs if/as needed
    """

    # Figure 1
    # Time newSort for different list lengths
    q = 2.8
    p = 100
    sizes = np.logspace(1, q, 15, dtype=int)
    times = np.zeros(len(sizes))

    # Initialize Figure
    fig1 = plt.figure(figsize=(10, 6))
    fig2 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)

    for i in range(3):
        for j, size in enumerate(sizes):
            # Set k
            if i == 0:
                # Merge sort
                k = 0
            elif i == 1:
                k = size/2
            else:
                k = size
            print(f'Figure 1, size={size}')

            # Precompute random lists so this is not timed
            X = np.random.randint(-1000, 1000, size=(p, size))

            # Time the algorithms over p runs
            t1 = time.time()
            for l in range(p):
                newSort(X[l, :], k=k)
            t2 = time.time()
            times[j] = (t2-t1)

        if i == 0:
            label = 'Merge Sort (k = 0)'
        elif i == 1:
            label = 'k = N/2'
        else:
            label = 'k = N'

        ax1.plot(sizes, times, marker='x', label=label)

        if i == 2:
            ax2.loglog(sizes, times, marker='x', label=label)
            ax2.loglog(sizes, sizes**2/sizes[len(sizes)//2]**2*times[len(sizes)//2], '--', label=r'$aN^2$')

    ax1.legend()
    ax1.set_title(f'Running time of {p} newSorts against N for different values of k')
    ax1.set_xlabel('N (size of list)')
    ax1.set_ylabel('Running time (s)')
    ax1.grid()
    ax2.legend()
    ax2.set_title(f'Running time for {p} merge sorts, k=0 against N')
    ax2.set_xlabel('N (size of list)')
    ax2.set_ylabel('Running time (s)')
    ax2.grid()

    # Figure 3
    # Time merge sort for different list lengths
    q = 5
    p = 100
    sizes = np.logspace(0, q, 20, dtype=int)
    times = np.zeros(len(sizes))

    # Initialize Figure
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    i = 2
    k = 0

    for j, size in enumerate(sizes):
        print('Figure 3')
        print(size, sizes[-1])
        # Precompute random lists so this is not timed
        X = (np.random.randint(-1000, 1000, size=(p, size)))

        # Time the algorithms over p runs
        t1 = time.time()
        for l in range(p):
            newSort(X[l, :], k=k)
        t2 = time.time()
        times[j] = (t2-t1)

    label = 'Merge Sort (k = 0)'

    ax3.loglog(sizes, times, marker='x', label=label)
    ax3.plot(sizes, sizes*np.log2(sizes)/(sizes[len(sizes)//2]*np.log2(sizes[len(sizes)//2]))*times[len(sizes)//2], \
        '--', label='aNlog(N)')

    ax3.legend()
    ax3.set_title(f'Running time for {p} merge sorts against N')
    ax3.set_xlabel('N (size of list)')
    ax3.set_ylabel('Running time (s)')
    ax3.grid()


    # Figure 4
    # Time newSort for different k
    q = 3
    p = 100
    sizes = [int(x) for x in [10**(q-1), 10**(q-0.5), 10**(q)]]
    k_list = np.logspace(1, q, 15, dtype=int)
    times = np.zeros(len(k_list))
    colors = ['r', 'g', 'b']

    # Initialize Figure
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)

    for i, size in enumerate(sizes):
        # Precompute random lists so this is not timed
        X = np.random.randint(-1000, 1000, size=(p, size))
        for j, k in enumerate(k_list):
            print(f'Figure 4, k={k}')
            # Time the algorithms over p runs
            t1 = time.time()
            for l in range(p):
                newSort(X[l, :], k=k)
            t2 = time.time()
            times[j] = (t2-t1)

        ax4.plot(k_list, times, marker='x', label=f'N={size}')

    ax4.legend()
    ax4.set_title(f'Running time of {p} newSorts against k for different values of N')
    ax4.set_xlabel('k')
    ax4.set_ylabel('Running time (s)')
    ax4.grid()


    # Figure 5
    # Time newSort for different k
    q = 50
    p = 100
    sizes = [100, 1000, 10000]
    k_list = np.linspace(1, q, q, dtype=int)
    times = np.zeros(len(k_list))
    colors = ['r', 'g', 'b']

    # Initialize Figure
    fig5 = plt.figure(figsize=(10, 6))
    ax5 = fig5.add_subplot(111)

    for i, size in enumerate(sizes):

        # Precompute random lists so this is not timed
        X = np.random.randint(-1000, 1000, size=(p, size))
        for j, k in enumerate(k_list):
            print(f'Figure 5, k={k}')

            # Time the algorithms over p runs
            t1 = time.time()
            for l in range(p):
                newSort(X[l, :], k=k)
            t2 = time.time()
            times[j] = (t2-t1)

        ax5.plot(k_list, times, marker='x', label=f'N={size}')

    ax5.legend()
    ax5.set_title(f'Running time of {p} newSorts against k for different values of N')
    ax5.set_xlabel('k')
    ax5.set_ylabel('Running time (s)')
    ax5.grid()

    plt.show()

    return None


def findTrough(L):
    """Find and return a location of a trough in L
    """

    # Set left and right indices
    istart = 0
    iend = len(L) - 1

    # Handle special cases
    if iend <= 0:
        return -(iend+2)

    # Modified iterative binary search
    while istart < iend:

        imid = int(0.5*(istart+iend))

        # Check middle two elements
        if L[imid] <= L[imid+1]:
            # Sequence is increasing or flat
            iend = imid
        else:
            # Sequence is decreasing
            istart = imid + 1
    return istart


if __name__=='__main__':
    inputs=None
    outputs=time_newSort(inputs)
