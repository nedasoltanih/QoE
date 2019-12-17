from math import sqrt
from scipy import spatial
import distance

"""calculate similarity between two objects
Params:
    r1 -- first user object
    r2 -- second user object
    d_type -- type of distance function to be used
    all_numeric -- indicates if the data is preprocessed and so all data types are numeric
Returns:
    float -- distance between two objects
"""


def calculate_distance(u1, u2, d_type, all_numeric, id_label, weights):
    w_all = weights['dist_weight'].copy()

    #if the data is preprocessed and all fields are converted to numeric
    if all_numeric:
        return {
            "jaccard": distance.jaccard(u1, u2),
            "euclidean": sqrt(sum(pow(weights * euclidean_dist(a, b), 2) for a, b in zip(u1[1:], u2[1:]))),
            "cosine": spatial.distance.cosine(u1, u2),
            "sorensen": distance.sorensen(u1, u2),
            "hamming": distance.hamming(u1, u2, normalized=True)
        }[d_type]

    # if all data is not numeric, distance for
    else:
        u1_nominal_columns = []
        u1_numeric_columns = []
        u2_nominal_columns = []
        u2_numeric_columns = []
        w_nominal = []
        w_numeric = []

        for i in range(len(u1)):
            if isinstance(u1[i], str):
                u1_nominal_columns.append(u1[i])
                u2_nominal_columns.append(u2[i])
                w_nominal.append(w_all[i])
            else:
                u1_numeric_columns.append(u1[i])
                u2_numeric_columns.append(u2[i])
                w_numeric.append(w_all[i])

        dist = 0
        if len(u1_nominal_columns) > 0:
            if d_type == "jaccard":
                dist = distance.jaccard(u1_nominal_columns, u2_nominal_columns)
            else:
                if d_type == "sorensen":
                    dist = distance.sorensen(u1_nominal_columns, u2_nominal_columns)
        dist += sqrt(sum(pow(w * euclidean_dist(a, b), 2) for a, b, w in zip(u1_numeric_columns, u2_numeric_columns, w_numeric)))
        return dist


"""Euclidean distance between two variables

Returns:
    float -- distance between two vairables
"""


def euclidean_dist(a, b):
    if isinstance(a, str):
        if a == b:
            return 0
        else:
            return 1
    else:
        return a-b
