"""This class is defined for link prediction based scenario
in this scenario networks are created for users and services
the goal is to predict link weight for a user-service connection

In this approach we find the shortest path between the user and the service

Version 3: network creation for each QoE parameter: Video quality, Impairment, Enjoyment, Satisfaction, Endurability, Involvement, ...
"""

import pandas as pd
import json
import os
import numpy as np
import distance_functions
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import networkx as nx
import communities
import config
import time

"""read files

Returns:
    dataframe -- file contents
"""


def read_file(data_file):
    with open(os.path.join(config.FULL_PATH + config.dataset_name, data_file + '.json')) as specs_file:
        specs = json.load(specs_file)
    if 'average_columns' in specs.keys():
        config.qtypes = specs['average_columns']
    return pd.read_csv(
        os.path.join(config.FULL_PATH + config.dataset_name, specs['file']),
        sep=specs['separator'],
        names=specs['attributes'],
        dtype={attribute_name: attribute_type for attribute_name,
               attribute_type in zip(specs['attributes'], specs['types'])},
    ), specs['id_label']


"""Preprocess ratings: get ratings for all QoE types and return average for one of them

Returns:
    [type] -- [description]
"""


def ratings_preprocess(data_file, ratings, type):
    # read ratings json file
    with open(os.path.join(config.FULL_PATH + config.dataset_name, data_file + '.json')) as specs_file:
        specs = json.load(specs_file)

    # needed columns to get average for QoE
    av_columns = specs['average_columns'][type]

    removable_cols = []
    for key in specs['average_columns']:
        for kkey in specs['average_columns'][key]:
            removable_cols.append(kkey)
    # other columns that must stay in ratings dataframe
    other_columns = set(specs['attributes']) - set(removable_cols)

    # new ratings dataframe which includes new QoE
    ratings_2 = pd.DataFrame(columns=[other_columns, 'QoE'])
    qoe = pd.Series(ratings[av_columns].mean(axis=1), name='QoE')
    ratings_2 = pd.concat([ratings[other_columns], qoe],  axis=1)

    # set satisfaction threshold based on QoE boundaries
    config.satisfaction_threshold = float(
        ((ratings_2.max(axis=0))['QoE']) / 2) + 0.5
    return ratings_2


"""preprocess data: encode nominal data to numerical, normalize data

Returns:
    dataframe -- processed data including only numeric and normalized data (label column is excluded)
"""


def convert_to_numeric(records, id_label):
    nominal_columns = []

    # label encoding to convert strings to numeric
    le = preprocessing.LabelEncoder()

    # Iterating over all the common columns in train and test
    for col in records.columns.values:

        # Encoding only categorical variables
        if records[col].dtypes == 'object':
            if col != id_label:
                nominal_columns.append(str(col))

                # Using whole data to form an exhaustive list of levels
                data = records[col].append(records[col])
                le.fit(data.values)
                records[col] = le.transform(records[col])

    # one-hot-encoder to convert nominal values to one-hot
    enc = preprocessing.OneHotEncoder(sparse=False)
    encoded_records = records.drop(labels=nominal_columns, axis=1, )

    for col in nominal_columns:
        data = records[[col]]
        enc.fit(data)

        # Fitting One Hot Encoding on train data
        temp = enc.transform(data)

        # Changing the encoded features into a data frame with new column names
        temp = pd.DataFrame(
            temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index])

        # adding the new One Hot Encoded varibales to the train data frame
        encoded_records = pd.concat([encoded_records, temp], axis=1)
    return encoded_records


"""Preprocess data and rescale numeric records to [0, 1]

Returns:
    dataframe -- rescaled records
"""


def preprocess_data(records, id_label):
    nominal_columns = []

    # Iterating over all the common columns
    for col in records.columns.values:

        # Finding categorical variables
        if records[col].dtypes == 'object':
            if col != id_label:
                nominal_columns.append(str(col))

    nominal_ = records[nominal_columns]
    id_ = records[id_label]
    numeric_ = (records.drop(labels=nominal_columns, axis=1, )
                ).drop(labels=id_label, axis=1, )

    if numeric_.empty:
        return records, nominal_columns

    else:
        # scaler to scale all values between 0 and 1
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        temp = scaler.fit_transform(numeric_)

        temp = pd.DataFrame(data=temp, index=range(
            0, len(temp)), columns=numeric_.columns)

        # concat all columns
        rescaled_records = pd.concat(
            [id_, temp, nominal_], axis=1)

        np.set_printoptions(precision=3)
        # print(rescaled_records[0:5][:])
        return rescaled_records, nominal_columns


"""connect data rows based on distance

Returns:
    dataframe -- including a csv of source, target, weight and label can be imported as network
"""


def connect(edge_lable, records, id_label, distance_function, dist_weight, nominal_columns):
    maxdist = 0
    edges = pd.DataFrame(columns=['source', 'target', 'weight', 'Label', ])

    # TODO: this part must be corrected and include weights here too
    if config.CONVERT_TO_NUMERIC:
        # if all data types is numeric
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                if i != j:
                    # excluding id from distance measure
                    dist = distance_functions.calculate_distance_numeric(
                        records.loc[i].drop(id_label), records.loc[j].drop(id_label), distance_function, dist_weight.dist_weight)
                    edges.loc[len(edges)] = [records[id_label][i],
                                             records[id_label][j], dist, edge_lable]
                    if dist > maxdist:
                        maxdist = dist

    else:
        # if data types are mixed
        numeric_columns = (set(records.columns) -
                           set(nominal_columns)) - {id_label}
        w_nominal = dist_weight.loc[dist_weight['attr_name'].isin(
            nominal_columns)]
        # & ~dist_weight['attr_name'].isin({id_label})]
        w_numeric = dist_weight.loc[~dist_weight['attr_name'].isin(
            nominal_columns)]

        for i in range(len(records)):
            for j in range(i+1, len(records)):
                # excluding id from distance measure
                dist = distance_functions.calculate_distance_mixed(
                    records.loc[i].drop(id_label), records.loc[j].drop(id_label), distance_function, nominal_columns, w_nominal, numeric_columns, w_numeric)
                edges.loc[len(edges)] = [records[id_label][i],
                                         records[id_label][j], dist, edge_lable]
                if dist > maxdist:
                    maxdist = dist

        return edges, maxdist


"""purge edges having low weights (lower than the threshold)

Returns:
    [dataframe] -- list of the edges
"""


def purge_edges(edges, distance_threshold, edge_lable, distance_function, alpha, maxdist):
    if edge_lable == 'u':
        filename = 'user_edges_' + distance_function + '_' + \
            str(distance_threshold) + '_' + str(alpha) + '.csv'
    else:
        filename = 'service_edges_' + distance_function + \
            '_' + str(distance_threshold) + '_' + str(alpha) + '.csv'

    for index, e in edges.iterrows():
        dist = e['weight']

        # normalize the distance to [0, 1]
        if maxdist > 0:
            dist = dist/maxdist

        # drop edges having more distance that threshold
        if dist > distance_threshold:
            edges.drop(index, inplace=True)
        else:
            #edges.loc[index,'weight'] = dist
            e['weight'] = dist

    edges.to_csv(os.path.join(
        config.FULL_PATH + config.dataset_name, filename), index=False)
    return edges


"""connect users to services based on rates they've given to services

Returns:
    dataframe -- including a csv of source, target, weight and label can be imported as network
"""


def connect_users_to_services(ratings, r_rate_label):
    edges = ratings.copy()
    edges.columns = ['source', 'target', 'weight']
    edges['Label'] = 'r'
    return edges


"""get QoE based on shortest path between two nodes

Returns:
    [float] -- the value of QoE
"""


def get_qoe(source, target, network, algorithm):
    G = nx.to_networkx_graph(
        network, create_using=None, multigraph_input=False)
    try:
        return nx.bellman_ford_path_length(G, source, target)
    except:
        return -1


"""main function
"""


def main():
    print("Dataset: ", config.dataset_name)
    print("Reading the data from files")

    # read from file
    users, config.u_id_label = read_file("users")
    services, config.s_id_label = read_file("videos")
    ratings, r_rate_label = read_file("u_ratings")
    r_rate_label = 'QoE'

    if config.CONVERT_TO_NUMERIC:
        print("Converting nomilnal columns to numeric")
        users = convert_to_numeric(users, config.u_id_label)
        services = convert_to_numeric(services, config.s_id_label)

    if config.PREPROCESS:
        print("Preprocessing the data")
        # preprocess
        users, u_nominal_columns = preprocess_data(users, config.u_id_label)
        services, s_nominal_columns = preprocess_data(
            services, config.s_id_label)
        users.to_csv(os.path.join(
            config.FULL_PATH + config.dataset_name, 'preprocessed_users.csv'), index=False)
        services.to_csv(os.path.join(
            config.FULL_PATH + config.dataset_name, 'preprocessed_services.csv'), index=False)

    print("Working on the data...")

    qoe_type = "General_positive_feelings"

    print("QoE Type = " + str(qoe_type))
    ratings_2 = ratings_preprocess("u_ratings", ratings, qoe_type)

    # connect users to services based on their rates
    processed_ratings = connect_users_to_services(ratings_2, r_rate_label)

    print("  Calculating weights")
    # get weights of columns for user and service
    dist_weight = pd.DataFrame(columns=['attr_name', 'dist_weight'])
    for attr in users.columns.drop(config.u_id_label):
        w = communities.get_attr_weight(
            users, services, ratings_2, attr, r_rate_label)
        dist_weight.loc[len(dist_weight)] = [attr, w]

    alpha = 1.5
    print("  Alpha = " + str(alpha))

    # compute QoE using "shortest_path" approach
    approach = "shortest_path"

    # distance function for nominal columns
    distance_function = "jaccard"
    print("  distance function = " + distance_function)

    # connect and create social network
    new_weights = pd.DataFrame(columns=['attr_name', 'dist_weight'])
    new_weights['attr_name'] = users.columns.drop(
        config.u_id_label)
    new_weights.loc[:, 'dist_weight'] *= alpha
    all_connected_users, max_u_dist = connect(
        'u', users, config.u_id_label, distance_function, new_weights, u_nominal_columns)

    service_weights = pd.DataFrame(
        columns=['attr_name', 'dist_weight'])
    service_weights['attr_name'] = services.columns.drop(
        config.s_id_label)
    service_weights['dist_weight'] = 1
    all_connected_services, max_s_dist = connect(
        's', services, config.s_id_label, distance_function, service_weights, s_nominal_columns)

    u_dist_threshold = 0.7
    u_dist_threshold = round(u_dist_threshold, 1)
    print("  User distance threshold = " + str(u_dist_threshold))

    connected_users = purge_edges(
        all_connected_users, u_dist_threshold, 'u', distance_function, alpha, max_u_dist)
    if len(connected_users) < 1:
        print("  User distance threshold is too high. Purged all edges for users!")
        return

    s_dist_threshold = 0.5
    s_dist_threshold = round(s_dist_threshold, 1)
    print("  Service distance threshold = " + str(s_dist_threshold))

    # create a file to write error and confusion matrix for each config
    result = pd.DataFrame(columns=['Alpha', 'User_distance_threshold', 'Service_distance_threshold', 'Distance_function',
                                   'Approach', 'Algorithm', 'TP', 'TN', 'FP', 'FN', 'MSE', 'MAE', 'Accuracy'])

    connected_services = purge_edges(
        all_connected_services, s_dist_threshold, 's', distance_function, alpha, max_s_dist)
    if len(connected_services) < 1:
        print("  Service distance threshold is too high. Purged all edges for services!")
        return

    # concat all networks
    network = pd.concat(
        [connected_users, connected_services])
    network = pd.concat([network, processed_ratings])
    network.to_csv(os.path.join(
        config.FULL_PATH + config.dataset_name, 'network_' + str(distance_function) + '_' + str(u_dist_threshold) + '_' + str(s_dist_threshold) + '_' + str(alpha) + '.csv'), index=False)

    start_time = time.time()

    # split data into test and train
    train_network, test_network = train_test_split(
        network, test_size=0.2)
    train_network.to_csv(os.path.join(
        config.FULL_PATH + config.dataset_name, 'train_network_' + str(distance_function) + '_' + str(u_dist_threshold) + '_' + str(s_dist_threshold) + '_' + str(alpha) + '.csv'), index=False)
    test_network.to_csv(os.path.join(
        config.FULL_PATH + config.dataset_name, 'test_network_' + str(distance_function) + '_' + str(u_dist_threshold) + '_' + str(s_dist_threshold) + '_' + str(alpha) + '.csv'), index=False)

    shortest_path_algorithm = "bellman_ford"
    print("  shortest path algorithm = " + shortest_path_algorithm)

    # Label is p which stands for predicted rating
    predicted_qoe = pd.DataFrame(
        columns=['source', 'target', 'Real_weight', 'Predicted_Weight', 'Label', ])

    # create confusion matrix
    CM = np.array([[0, 0], [0, 0]])

    for i in range(len(test_network)):
        if test_network['Label'].values[i] == 'r':
            user = test_network['source'].values[i]
            service = test_network['target'].values[i]
            rate = test_network['weight'].values[i]

            qoe = get_qoe(user, service, train_network,
                          shortest_path_algorithm)

            if qoe != -1 and rate != -1:
                predicted_qoe.loc[len(predicted_qoe)] = [
                    user, service, rate, qoe, 'p']

                # Add up TN, TP, FN, FP
                if rate > config.satisfaction_threshold:
                    if qoe > config.satisfaction_threshold:
                        CM[0][0] += 1
                    else:
                        CM[1][0] += 1
                else:
                    if qoe > config.satisfaction_threshold:
                        CM[0][1] += 1
                    else:
                        CM[1][1] += 1

    # compute error values and write in file
    if not predicted_qoe.empty:
        MSE = mean_squared_error(
            predicted_qoe['Real_weight'].values, predicted_qoe['Predicted_Weight'].values)
        MAE = mean_absolute_error(
            predicted_qoe['Real_weight'].values, predicted_qoe['Predicted_Weight'].values)
        predicted_qoe.loc[len(predicted_qoe)] = [
            'mean squared error:', MSE, '', '', 'error']
        predicted_qoe.loc[len(predicted_qoe)] = [
            'mean absolute error:', MAE, '', '', 'error']
        predicted_qoe.to_csv(os.path.join(config.FULL_PATH + config.dataset_name,
                                          'output_' + qoe_type + '_' +
                                          str(u_dist_threshold) + '_' +
                                          str(s_dist_threshold) + '_' +
                                          distance_function + '_'
                                          + str(alpha) + '_'
                                          + approach + '_' + shortest_path_algorithm + '.csv'), index=False)

        print("  MAE = ", str(MAE))
        print("  MSE = ", str(MSE))
        Acc = (CM[0][0]+CM[1][1])/(CM[0][0] +
                                   CM[1][1] + CM[0][1] + CM[1][0])
        print("  Accuracy = ", str(Acc))

        result.loc[len(result)] = [alpha, u_dist_threshold, s_dist_threshold, distance_function, approach, shortest_path_algorithm,
                                   CM[0][0], CM[1][1], CM[0][1], CM[1][0], MSE, MAE, str(Acc)]
        elapsed_time = time.time() - start_time
        print("  Time:", elapsed_time)

    with open(config.FULL_PATH + config.dataset_name + '/result_' + qoe_type + '.csv', 'a+') as f:
        result.to_csv(f, header=False)


if __name__ == '__main__':
    main()
