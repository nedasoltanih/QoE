from math import sqrt
from scipy import spatial
import distance
import config

"""calculate similarity between two objects
Params:
    u1 -- first object
    u2 -- second object
    d_type -- type of distance function to be used
    all_numeric -- indicates if the data is preprocessed and so all data types are numeric
    weights -- weights of each field
Returns:
    float -- distance between two objects
"""

def calculate_distance_numeric(u1, u2, d_type, weights):
    # if the data is preprocessed and all fields are converted to numeric
    return {
        "jaccard": distance.jaccard(u1, u2),
        "euclidean": sqrt(sum(pow((1/w) * (a-b), 2) for a, b, w in zip(u1, u2, weights))),
        "cosine": spatial.distance.cosine(u1, u2),
        "sorensen": distance.sorensen(u1, u2),
        "hamming": distance.hamming(u1, u2, normalized=True)
    }[d_type]

def calculate_distance_mixed(u1, u2, d_type, nominal_columns, w_nominal, numeric_columns, w_numeric):

    # if all data is not numeric, distance for
    dist = 0
    if len(nominal_columns) > 0:
        if d_type == "jaccard":
            dist = jaccard(u1[nominal_columns], u2[nominal_columns], w_nominal)
        else:
            if d_type == "sorensen":
                dist = sorensen(
                    u1[nominal_columns], u2[nominal_columns], w_nominal)
            else: # dtype == hamming:
                dist = hamming(u1[nominal_columns], u2[nominal_columns], w_nominal)
    dist += euclidean(u1[numeric_columns], u2[numeric_columns], w_numeric)
    if dist < 0:
        dist *= -1
    return dist


def euclidean(seq1, seq2, weights):
    dist = 0
    for i in range(0,len(seq1)):
        w = weights.loc[weights['attr_name'] == seq1.keys()[i],'dist_weight'].values[0]
        dist += pow((1/w) * (seq1[i] - seq2[i]), 2)
    return dist


def hamming(seq1, seq2, weights):
    """Compute the Hamming distance between the two sequences `seq1` and `seq2`.
    The Hamming distance is the number of differing items in two ordered
    sequences of the same length. If the sequences submitted do not have the
    same length, an error will be raised.

    If `normalized` evaluates to `False`, the return value will be an integer
    between 0 and the length of the sequences provided, edge values included;
    otherwise, it will be a float between 0 and 1 included, where 0 means
    equal, and 1 totally different. Normalized hamming distance is computed as:

        0.0                         if len(seq1) == 0
        hamming_dist / len(seq1)    otherwise
    """
    L = len(seq1)
    if L != len(seq2):
        raise ValueError("expected two strings of the same length")
    if L == 0:
        return 0.0
    dist = 0.0
    for i in range(0,len(seq1)):
        if seq1[i] != seq2[i]:
            dist += weights.loc[weights['attr_name'] == seq1.keys()[i],'dist_weight'].values[0]
    return dist

def jaccard(seq1, seq2, weights):
    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.

    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    L = 0.0
    for i in range(0,len(seq1)):
        if seq1[i] == seq2[i]:
            L += weights.loc[weights['attr_name'] == seq1.keys()[i],'dist_weight'].values[0]
    return 1 - (L / float(len(set1 | set2)))


def sorensen(seq1, seq2, weights):
    """Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """

    set1, set2 = set(seq1), set(seq2)
    L = 0.0
    for i in range(0,len(seq1)):
        if seq1[i] == seq2[i]:
            L += weights.loc[weights['attr_name'] == seq1.keys()[i],'dist_weight'].values[0]
    return 1 - (2 * L / float(len(set1) + len(set2)))