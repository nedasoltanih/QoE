import pandas
import config

def get_number_of_distinct_services(users, services, ratings, r_rate_label):
    i=0 # W(c)
    j=0 # Nr(c)
    for index, s in services.iterrows():
        temp = ratings.loc[ratings[config.u_id_label].isin(users[config.u_id_label])]
        temp = temp.loc[temp[r_rate_label] > config.satisfaction_threshold]
        temp = temp.loc[temp[config.s_id_label] == s[config.s_id_label]]
        i = i + len(temp) # p(r) for current service
        j = j + (1 if len(temp) > 0 else 0)
    return i, j

def get_attr_weight(users, services, ratings, attr, r_rate_label):
    values = users.groupby([attr])
    CSD = pandas.DataFrame(columns=['csd'])
    for v in values:
        group = users.loc[users[attr] == v[0]]
        NUC = len(group) # Nu(c)
        if NUC > 1:
            WC, NRC = get_number_of_distinct_services(group, services, ratings, r_rate_label)
            if NRC > 0 and NUC > 1:
                CSD.loc[len(CSD)] = [((WC/NRC) - 1) / (NUC - 1)]
    weight = 1 if CSD.empty else CSD['csd'].mean()
    return weight