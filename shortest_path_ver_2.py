"""This class is defined for link prediction based scenario
in this scenario networks are created for users and services
the goal is to predict link weight for a user-service connection

"""
import pandas
import json
import os
import numpy
import distance_functions
import rating_based
import similarity_based
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import networkx as nx
import communities

FULL_PATH = "/my_env/"
PREPROCESS = False

"""configuration class
"""
class config:
    dataset_name = "iQoE"#"CP_QAE_I" 
    u_id_label = ""
    s_id_label = ""


"""read files

Returns:
    dataframe -- file contents
"""
def read_file(data_file):
    with open(os.path.join(FULL_PATH + config.dataset_name, data_file + '.json')) as specs_file:
        specs = json.load(specs_file)
    return pandas.read_csv(
        os.path.join(FULL_PATH + config.dataset_name, specs['file']),
        sep=specs['separator'],
        names=specs['attributes'],
        dtype={attribute_name: attribute_type for attribute_name,
               attribute_type in zip(specs['attributes'], specs['types'])},
    ), specs['id_label']


"""preprocess data: encode nominal data to numerical, normalize data

Returns:
    dataframe -- processed data including only numeric and normalized data (label column is excluded)
"""


def preprocess_data(records, id_label):
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
        temp = pandas.DataFrame(
            temp, columns=[(col + "_" + str(i)) for i in data[col].value_counts().index])

        # adding the new One Hot Encoded varibales to the train data frame
        encoded_records = pandas.concat([encoded_records, temp], axis=1)

    # scaler to scale all values between 0 and 1
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    temp = encoded_records.drop(labels=str(id_label), axis=1, )
    rescaled_records = scaler.fit_transform(temp)

    # concat with ID's
    rescaled_records = pandas.DataFrame(rescaled_records, columns=temp.columns)
    rescaled_records = pandas.concat(
        [encoded_records[id_label], rescaled_records], axis=1)

    numpy.set_printoptions(precision=3)
    print(rescaled_records[0:5][:])
    return rescaled_records


"""connect data rows based on similarity

Returns:
    dataframe -- including a csv of source, target, weight and label can be imported as network
"""


def connect(r_type, records, id_label, distance_function, dist_weight, similarity_threshold=0):
    edges = pandas.DataFrame(columns=['source', 'target', 'weight', 'Label', ])
    if r_type:
        filename = 'user_edges_' + distance_function + '_' + \
            str(round(similarity_threshold, 1)) + '.csv'
        edge_lable = 'u'
    else:
        filename = 'service_edges_' + distance_function + \
            '_' + str(round(similarity_threshold, 1)) + '.csv'
        edge_lable = 's'

    # find the id index in order to ignore in distance
    id_indx = 0
    for col in records.columns:
        id_indx += 1
        if col == id_label:
            break

    for i in range(len(records)):
        for j in range(i+1, len(records)):
            if i != j:
                # 1: for excluding id from similarity measure
                dist = distance_functions.calculate_distance(
                    records.ix[i][id_indx:], records.ix[j][id_indx:], distance_function, PREPROCESS, id_label, dist_weight)
                if dist == 0:
                    sim = 2
                else:
                    if dist == 1:
                        sim = 0
                    else:
                        sim = 2/dist
                if sim > similarity_threshold:
                    edges.loc[len(edges)] = [records[id_label][i], records[id_label][j], sim, edge_lable]

    edges.to_csv(os.path.join(FULL_PATH + config.dataset_name, filename), index=False)
    print(edges[0:5][:])
    return edges


"""connect users to services based on rates they've given to services

Returns:
    dataframe -- including a csv of source, target, weight and label can be imported as network
"""


def connect_users_to_services(u_id_label, s_id_label, ratings, r_rate_label):
    edges = pandas.DataFrame(columns=['source', 'target', 'weight', 'Label', ])
    for i in range(len(ratings)):
        edges.loc[len(edges)] = [ratings[u_id_label][i],
                                 ratings[s_id_label][i], ratings[r_rate_label][i], 'r']
    edges.to_csv(os.path.join(FULL_PATH + config.dataset_name,
                              'ratings_edges.csv'), index=False)
    return edges


# in this approach we find the shortest path between the user and the service
def get_qoe(source, target, network, algorithm):
    G = nx.to_networkx_graph(
        network, create_using=None, multigraph_input=False)
    return {
        "dijkstra": nx.dijkstra_path_length(G, source, target),
        "bellman_ford": nx.bellman_ford_path_length(G, source, target)
    }[algorithm]


"""this function sorts a datafram descending based on weight

Returns:
    dataframe -- sorted list
"""		


def get_closest(r, records):
    d1 = records.loc[(records['source'] == r)]
    d2 = records.loc[(records['target'] == r)]
    frames = [d1, d2]
    connected = pandas.concat(frames)
    closest = connected.loc[(connected['weight'] == connected['weight'].max())]
    if closest.empty:
        return "", -1
    return closest['target'].values[0] if closest['source'].values[0] == r else closest['source'].values[0]


"""first scenario's main function
"""


def main():

    # read from file
    users, config.u_id_label = read_file("users")
    services, config.s_id_label = read_file("videos")
    ratings, r_rate_label = read_file("ratings")

    # get weights of columns for user and service
    dist_weight = pandas.DataFrame(columns=['attr_name', 'dist_weight'])
    for attr in users.columns.drop(config.u_id_label):
        dist_weight.loc[len(dist_weight)] = [attr, communities.get_attr_weight(users, services, ratings, attr, config.u_id_label, config.s_id_label, r_rate_label)]


    if PREPROCESS:
        # preprocess
        preprocessed_users = preprocess_data(users, config.u_id_label)
        preprocessed_services = preprocess_data(services, config.s_id_label)
        preprocessed_users.to_csv(os.path.join(
            FULL_PATH + config.dataset_name, 'preprocessed_users.csv'), index=False)
        preprocessed_services.to_csv(os.path.join(
            FULL_PATH + config.dataset_name, 'preprocessed_services.csv'), index=False)

    else:
        preprocessed_users = users
        preprocessed_services = services


    # create a file to write error for each config
    errors = pandas.DataFrame(columns=['user_similarity_threshold', 'service_similarity_threshold', 'distance_function',
                                       'approach', 'algorithm', 'MSE', 'MAE'])

    # compute QoE using "similarity_based" approach
    approach = "shortest_path"
    for u_sim_threshold in numpy.arange(0.7, 1.0, 0.2):
        for s_sim_threshold in numpy.arange(0.7, 1.0, 0.2):

            #distance function for nominal columns
            for distance_function in ["jaccard"]:#, "sorensen", "cosine", "jaccard", "euclidean", "hamming"]:

                # connect and create social network
                connected_users = connect(
                    True, preprocessed_users, config.u_id_label, distance_function, dist_weight, u_sim_threshold)
                service_weights = pandas.DataFrame(columns=['attr_name', 'dist_weight'])
                service_weights['attr_name'] = services.columns.drop(config.s_id_label)
                service_weights['dist_weight'] = 1

                connected_services = connect(
                    False, preprocessed_services, config.s_id_label, distance_function, service_weights, s_sim_threshold)

                # connect users to services based on their rates
                processed_ratings = connect_users_to_services(
                    config.u_id_label, config.s_id_label, ratings, r_rate_label)

                # concat all networks
                network = pandas.concat(
                    [connected_users, connected_services])
                network = pandas.concat([network, processed_ratings])

                # split data into test and train
                train_network, test_network = train_test_split(
                    network, test_size=0.2)
                train_network.to_csv(os.path.join(
                    FULL_PATH + config.dataset_name, 'train_network_' + str(distance_function) + '_' + str(u_sim_threshold) + '_' + str(s_sim_threshold) + '.csv'), index=False)
                test_network.to_csv(os.path.join(
                    FULL_PATH + config.dataset_name, 'test_network_' + str(distance_function) + '_' + str(u_sim_threshold) + '_' + str(s_sim_threshold) + '.csv'), index=False)
                print(train_network[0:5][:])
                print(test_network[0:5][:])

                for shortest_path_algorithm in ["bellman_ford"]: #"dijkstra", "bellman_ford"
                    # for CP_QAE rate is 1 to 5, for iQoE it is 1 to 7
                    # Label is p which stands for predicted rating
                    predicted_qoe = pandas.DataFrame(
                        columns=['source', 'target', 'Real_weight', 'Predicted_Weight', 'Label', ])
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
                        predicted_qoe.to_csv(os.path.join(FULL_PATH + config.dataset_name,
                                                        'output_' + '_' +
                                                        str(u_sim_threshold) + '_' +
                                                        str(s_sim_threshold) + '_' +
                                                        distance_function + '_'
                                                        + approach + '_' + shortest_path_algorithm + '.csv'), index=False)
                        print(predicted_qoe[0:5][:])

                    errors.loc[len(errors)] = [
                        u_sim_threshold, s_sim_threshold, distance_function,
                        approach, shortest_path_algorithm, MSE, MAE]
                    print(errors)

        print(errors[0:5][:])

    with open(FULL_PATH + config.dataset_name + '/errors.csv', 'a+') as f:
        errors.to_csv(f, header=False)


if __name__ == '__main__':
    main()
