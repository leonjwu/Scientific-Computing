""" Your college id here: 01190736
    Template code for part 2, contains 3 functions:
    codonToAA: returns amino acid corresponding to input amino acid
    DNAtoAA: to be completed for part 2.1
    pairSearch: to be completed for part 2.2
"""


def codonToAA(codon):
	"""Return amino acid corresponding to input codon.
	Assumes valid codon has been provided as input
	"_" is returned for valid codons that do not
	correspond to amino acids.
	"""
	table = {
		'ATA':'i', 'ATC':'i', 'ATT':'i', 'ATG':'m',
		'ACA':'t', 'ACC':'t', 'ACG':'t', 'ACT':'t',
		'AAC':'n', 'AAT':'n', 'AAA':'k', 'AAG':'k',
		'AGC':'s', 'AGT':'s', 'AGA':'r', 'AGG':'r',
		'CTA':'l', 'CTC':'l', 'CTG':'l', 'CTT':'l',
		'CCA':'p', 'CCC':'p', 'CCG':'p', 'CCT':'p',
		'CAC':'h', 'CAT':'h', 'CAA':'q', 'CAG':'q',
		'CGA':'r', 'CGC':'r', 'CGG':'r', 'CGT':'r',
		'GTA':'v', 'GTC':'v', 'GTG':'v', 'GTT':'v',
		'GCA':'a', 'GCC':'a', 'GCG':'a', 'GCT':'a',
		'GAC':'d', 'GAT':'d', 'GAA':'e', 'GAG':'e',
		'GGA':'g', 'GGC':'g', 'GGG':'g', 'GGT':'g',
		'TCA':'s', 'TCC':'s', 'TCG':'s', 'TCT':'s',
		'TTC':'f', 'TTT':'f', 'TTA':'l', 'TTG':'l',
		'TAC':'y', 'TAT':'y', 'TAA':'_', 'TAG':'_',
		'TGC':'c', 'TGT':'c', 'TGA':'_', 'TGG':'w',
	}
	return table[codon]


def DNAtoAA(S):
    """Convert genetic sequence contained in input string, S,
    into string of amino acids corresponding to the distinct
    amino acids found in S and listed in the order that
    they appear in S
    """

    AA = ""
    AA_set = set()
    n = len(S)
    # Loop through sequence in steps of 3
    for i in range(0, n, 3):
        a = codonToAA(S[i:i+3])
        # Check whether amino acid is already in AA
        if a not in AA_set:
            AA_set.add(a)
            AA += a

    return AA


def char2base4(S):
    """ Convert gene sequence to base 4 string
    """
    c2b = {}
    c2b['A'] = '0'
    c2b['C'] = '1'
    c2b['G'] = '2'
    c2b['T'] = '3'

    L = ''
    for s in S:
        L += c2b[s]
    return L


def hash10(S, base):
    """Convert list S to base-10 number where
    base specifies the base of S
    """
    f = 0
    for s in S[:-1]:
        f = base*(int(s)+f)
    f += int(S[-1])
    return f


def pairSearch(L,pairs):
    """Find locations within adjacent strings (contained in input list,L)
    that match k-mer pairs found in input list pairs. Each element of pairs
    is a 2-element tuple containing k-mer strings
    """

    # Convert L to base 4 (Note that the input is being modified)
    n = len(L)
    for i in range(n):
        L[i] = char2base4(L[i])

    # Convert each pairs to base 4
    pairs4 = []
    n = len(pairs)
    for i in range(n):
        pairs4.append((char2base4(pairs[i][0]), char2base4(pairs[i][1])))

    # Create a dictionary, with keys as hashes of the first of the pairs,
    # and values as lists of tuples, with the second of the pairs with its index in pairs
    pairs_hash_dict = {}
    for index, pair in enumerate(pairs4):
        # Calculate hash of the base 4 k-mers in pairs
        hashed0 = hash10(pair[0], 4)
        hashed1 = hash10(pair[1], 4)
        if hashed0 in pairs_hash_dict:
            if hashed1 in pairs_hash_dict[hashed0]:
                pairs_hash_dict[hashed0][hashed1].append(index)
            else:
                pairs_hash_dict[hashed0].update({hashed1: [index]})
        else:
            pairs_hash_dict[hashed0] = {hashed1: [index]}


    # Initialize
    k = len(pairs[0][0])
    N = len(L[0])  # Assumes all l in L are of length N
    locations = []
    first_digit = [int(l[0]) for l in L]

    # Compute first k-mer hash for all sequences
    hashes = [hash10(l[:k], 4) for l in L]

    # Main loop across base 4 strings in l for loop below
    for j in range(k-1, N):

        # Loop across l in L
        match = False
        for i, l in enumerate(L):

            # Separate case for first step
            if j != k-1:
                # Complete the rolling hash calculation
                hashes[i] = 4 * (hashes[i] - first_digit[i]*4**(k-1)) + int(l[j])

                # Store first base 4 digit to subtract at next iteration
                first_digit[i] = int(l[j-k+1])

            # If previous sequence matched with first of a pair
            if match:
                # Check for a match in this sequence
                if hashes[i] in pairs_hash_dict[match]:
                    index_list = pairs_hash_dict[match][hashes[i]]
                    # Below loop accounts for EXACT duplicates in pairs
                    for index in index_list:
                        locations.append([j-k+1, i-1, index])

            # Check if k-mer exists in first of pairs in pairs
            if hashes[i] in pairs_hash_dict:
                # Check pair2 in next sequence
                match = hashes[i]
            else:
                match = False

    return locations
