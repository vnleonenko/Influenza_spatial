import pandas as pd
import numpy as np
from scipy import stats
import warnings
import datetime
import json
import os
import matplotlib.pyplot as plt
from statistics import mean
from functools import partial
from collections import defaultdict
import multiprocessing as mp
warnings.filterwarnings('ignore')
pd.options.display.width = None
pd.options.display.max_columns = None

data = pd.read_csv(r'~/PyCharmProjects/influenza/sp5-15km/people.txt', sep='\t', index_col=0)
data = data[['sp_id', 'sp_hh_id', 'age', 'sex', 'work_id']]
households = pd.read_csv(r'~/PyCharmProjects/influenza/sp5-15km/households.txt', sep='\t')
households = households[['sp_id', 'latitude', 'longitude']]
dict_school_id = json.load(open(os.path.expanduser(r'~/PyCharmProjects/influenza/sp5-15km/school_id_5-15km.json')))
dict_school_len = [len(dict_school_id[i]) for i in dict_school_id.keys()]

infected = 10
alpha = 0.17
lmbd = 0.7
number = 5
days = range(1, 81)


def func_b_r(inf_day):
    a = [0.0, 0.0, 0.9, 0.9, 0.55, 0.3, 0.15, 0.05]
    if inf_day < 9:
        return a[inf_day - 1]
    else:
        return 0


def main_function(number_seed, dataset):
    np.random.seed(number_seed)
    y = np.random.choice(np.array(dataset[dataset.susceptible == 1].sp_id), infected, replace=False)
    data_susceptible = dataset[dataset.susceptible == 1]
    data_susceptible.loc[pd.np.in1d(data_susceptible.sp_id, y), ['infected', 'susceptible', 'illness_day']] = [1, 0, 3]

    print(data_susceptible[data_susceptible.infected == 1].drop_duplicates())
    id_susceptible_list, latitude_list, longitude_list, type_list, id_place_list, days_inf, results = [], [], [], [], [], [], []

    dict_hh_id = dict_school_id.copy()
    dict_hh_id.clear()
    dict_hh_id = {i: list(data_susceptible[(data_susceptible.sp_hh_id == i) & (data_susceptible.susceptible == 1)].index)
                  for i in data_susceptible.loc[data_susceptible.infected == 1, 'sp_hh_id']}

    dict_work_id = dict_school_id.copy()
    dict_work_id.clear()
    dict_work_id = {int(i): list(data_susceptible[(data_susceptible.age > 17) & (data_susceptible.work_id == i) & (data_susceptible.susceptible == 1)].index)
                    for i in data_susceptible.loc[(data_susceptible.infected == 1) & (data_susceptible.age > 17) & (data_susceptible.work_id != 'X'), 'work_id']}

    [dict_school_id[str(i)].remove(j) for i, j in zip(data_susceptible.loc[(data_susceptible.infected == 1) & (data_susceptible.age <= 17) &
                                                                           (data_susceptible.work_id != 'X'), 'work_id'],
                                                      data_susceptible[(data_susceptible.infected == 1) & (data_susceptible.age <= 17) &
                                                                       (data_susceptible.work_id != 'X')].index)]
    vfunc_b_r = np.vectorize(func_b_r)

    for j in days:
        if len(data_susceptible[data_susceptible.illness_day > 2]) != 0:
            x_rand = np.random.rand(1000000)
            curr = data_susceptible[data_susceptible.infected == 1]
            hh_inf, work_inf, school_inf = defaultdict(list), defaultdict(list), defaultdict(list)
            for _, row in curr.iterrows():
                ill_day = row.illness_day
                if ill_day > 2:
                    hh_inf[row.sp_hh_id].append(ill_day)
                    if row.work_id != 'X':
                        if row.age > 17:
                            work_inf[row.work_id].append(ill_day)
                        else:
                            school_inf[row.work_id].append(ill_day)

            real_inf_hh = np.array([])
            for i in hh_inf:
                if i not in dict_hh_id.keys():
                    dict_hh_id.update({i: list(data_susceptible[(data_susceptible.sp_hh_id == i) & (data_susceptible.susceptible == 1)].index)})
                hh_len = len(dict_hh_id[i])
                if hh_len != 0:
                    temp = vfunc_b_r(hh_inf[i])
                    prob = np.repeat(temp, hh_len) * lmbd
                    curr_length = len(prob)
                    hh_rand = x_rand[:curr_length]
                    x_rand = x_rand[curr_length:]
                    real_inf = len(hh_rand[hh_rand < prob])
                    if hh_len < real_inf:
                        real_inf = hh_len
                    real_inf_id = np.random.choice(np.array(dict_hh_id[i]), real_inf, replace=False)
                    real_inf_hh = np.concatenate((real_inf_hh, real_inf_id))

                    id_susceptible_list.extend(data_susceptible.sp_id[real_inf_id])
                    type_list.extend(['household'] * len(real_inf_id))
                    id_place_list.extend(data_susceptible.sp_hh_id[real_inf_id])
                    days_inf.extend([j] * len(real_inf_id))
            print(number_seed, 'households', len(hh_inf), datetime.datetime.now())

            real_inf_work = np.array([])
            some_current = data_susceptible[(data_susceptible.work_id != 'X') & (data_susceptible.age > 17) & (data_susceptible.susceptible == 1)]
            some_current[['work_id']] = some_current[['work_id']].astype(int)
            for i in work_inf:
                if i not in dict_work_id.keys():
                    dict_work_id.update({i: list(some_current[some_current.work_id == int(i)].index)})
                work_len = len(dict_work_id[i])
                if work_len != 0:
                    temp = vfunc_b_r(work_inf[i])
                    prob = np.repeat(temp, work_len) * lmbd
                    curr_length = len(prob)
                    work_rand = x_rand[:curr_length]
                    x_rand = x_rand[curr_length:]
                    real_inf = len(work_rand[work_rand < prob])
                    if work_len < real_inf:
                        real_inf = work_len
                    real_inf_id = np.random.choice(np.array(dict_work_id[i]), real_inf, replace=False)
                    real_inf_work = np.concatenate((real_inf_work, real_inf_id))

                    id_susceptible_list.extend(data_susceptible.sp_id[real_inf_id])
                    type_list.extend(['workplace'] * len(real_inf_id))
                    id_place_list.extend(map(lambda x: int(x), data_susceptible.work_id[real_inf_id]))
                    days_inf.extend([j] * len(real_inf_id))
            print(number_seed, 'workplaces', len(work_inf), datetime.datetime.now())

            real_inf_school = np.array([])
            for i in school_inf:
                school_len = len(dict_school_id[str(i)])
                if school_len != 0:
                    length = dict_school_len[list(dict_school_id.keys()).index(str(i))]
                    temp = vfunc_b_r(school_inf[i])
                    prob_cont = 8.5 / (length - 1) if (8.5 + 1) < length else 1
                    res = np.prod(1 - prob_cont * lmbd * temp)
                    real_inf = np.random.binomial(length - 1, 1 - res)
                    if school_len < real_inf:
                        real_inf = school_len
                    real_inf_id = np.random.choice(np.array(dict_school_id[str(i)]), real_inf, replace=False)
                    real_inf_school = np.concatenate((real_inf_school, real_inf_id))

                    id_susceptible_list.extend(data_susceptible.sp_id[real_inf_id])
                    type_list.extend(['school'] * len(real_inf_id))
                    id_place_list.extend(map(lambda x: int(x), data_susceptible.work_id[real_inf_id]))
                    days_inf.extend([j] * len(real_inf_id))
            print(number_seed, 'schools', len(school_inf), datetime.datetime.now())

            real_inf = np.concatenate((real_inf_hh, real_inf_school, real_inf_work))
            real_inf = np.unique(real_inf.astype(int))
            data_susceptible.loc[real_inf, ['infected', 'illness_day', 'susceptible']] = [1, 1, 0]

            current_hh_id = []
            [current_hh_id.extend(i) for i in dict_hh_id.values()]
            check_id = [True if i in current_hh_id else False for i in real_inf]
            check_id = [i for i, x in enumerate(check_id) if x is True]
            inf_hh = real_inf[check_id]
            [dict_hh_id[i].remove(j) for i, j in zip(data_susceptible.loc[inf_hh, 'sp_hh_id'], inf_hh)]

            current_wp_id = []
            [current_wp_id.extend(i) for i in dict_work_id.values()]
            check_id = [True if i in current_wp_id else False for i in real_inf]
            check_id = [i for i, x in enumerate(check_id) if x is True]
            inf_wp = real_inf[check_id]
            [dict_work_id[i].remove(j) for i, j in zip(data_susceptible.loc[inf_wp, 'work_id'], inf_wp)]

            inf_school = real_inf[(data_susceptible.loc[real_inf, 'work_id'] != 'X') & (data_susceptible.loc[real_inf, 'age'] <= 17)]
            [dict_school_id[str(i)].remove(j) for i, j in zip(data_susceptible.loc[inf_school, 'work_id'], inf_school)]

        newly_infected = len(data_susceptible[data_susceptible.illness_day == 1])
        results.append(newly_infected)

        dataset_infected = data_susceptible[data_susceptible.infected == 1][['sp_id', 'sp_hh_id']]
        dataset_infected = dataset_infected.sort_values(by=['sp_hh_id'])
        temp = households.loc[households.index.intersection(dataset_infected.sp_hh_id), ['latitude', 'longitude']]
        temp.index = dataset_infected.index
        dataset_infected['latitude'] = temp.latitude
        dataset_infected['longitude'] = temp.longitude
        dataset_infected[['sp_id', 'latitude', 'longitude']].to_csv(
            r'~/PyCharmProjects/influenza/results/sp5-15km_apart+work/infected_{}_seed_{}_day.txt'.format(number_seed, j), sep='\t', index=False)

        data_susceptible.loc[data_susceptible.infected == 1, 'illness_day'] += 1
        data_susceptible.loc[data_susceptible.illness_day > 8, ['infected', 'illness_day']] = 0

        print(number_seed, j, newly_infected, int(data_susceptible[['infected']].sum()),
              int(data_susceptible[['susceptible']].sum()), datetime.datetime.now())
        print()
    dataset_place = pd.DataFrame({'day': days_inf, 'sp_id': id_susceptible_list, 'place_type': type_list, 'place_id': id_place_list})
    dataset_place.to_csv(r'~/PyCharmProjects/influenza/results/sp5-15km_apart+work/{}_newly_infected_place.txt'.format(number_seed),
                         sep='\t', index=False)
    return results


if __name__ == '__main__':
    data[['sp_id', 'sp_hh_id', 'age']] = data[['sp_id', 'sp_hh_id', 'age']].astype(int)
    data[['work_id']] = data[['work_id']].astype(str)
    data = data.sample(frac=1)
    households[['sp_id']] = households[['sp_id']].astype(int)
    households[['latitude', 'longitude']] = households[['latitude', 'longitude']].astype(float)
    households.index = households.sp_id
    data['susceptible'] = 0
    data['infected'] = 0
    data['illness_day'] = 0

    data.loc[np.random.choice(data.index, round(len(data) * alpha), replace=False), 'susceptible'] = 1
    [dict_school_id[str(i)].remove(j) for i, j in zip(data.loc[(data.susceptible == 0) & (data.age <= 17) & (data.work_id != 'X'), 'work_id'],
                                                      data[(data.susceptible == 0) & (data.age <= 17) & (data.work_id != 'X')].index)]
    print(sum(data['susceptible']), round(sum(data['susceptible']) / len(data), 4), lmbd)

    for l in range(number):
        np.random.seed(l)
        print(np.random.choice(np.array(data[data['susceptible'] == 1].sp_id), infected, replace=False))
    print()

    with mp.Pool(number) as pool:
        output = pool.map(partial(main_function, dataset=data), range(number))
    print(output)

    mean_ = [*map(mean, zip(*output))]
    print(mean_)
    print([*map(min, zip(*output))])
    print([*map(max, zip(*output))])

    plt.plot(days, mean_, color='red')
    plt.plot(days, [*map(min, zip(*output))], color='green')
    plt.plot(days, [*map(max, zip(*output))], color='green')
    plt.legend(('Mean', 'Min', 'Max'), loc='upper right')
    plt.xlabel('Duration of days')
    plt.ylabel('Active incidence cases')
    plt.show()
