__author__ = '@P4ndaM1x, Michał Rutkowski'

import numpy as np


def get_incomplete_data_keys(data: dict, ind_labels: dict, subind_labels: dict) -> set:
    incomplete_data_keys = []
    for date, indicators in data.items():
        if indicators is None or indicators.keys() != ind_labels.keys():
            incomplete_data_keys.append(date)
            continue
        for indicator_id, subindicators in indicators.items():
            if subindicators is None or subindicators.keys() != subind_labels[indicator_id].keys():
                incomplete_data_keys.append(date)
                break
            for subindicator_id, value in subindicators.items():
                if value is None:
                    incomplete_data_keys.append(date)
                    break
    return set(incomplete_data_keys)


def filter_dictionary_data(data: dict, keys_to_remove: set):
    new_data = data.copy()
    for key in keys_to_remove.intersection(set(new_data.keys())):
        del new_data[key]
    return new_data


if __name__ == "__main__":
    normalized_data = dict(np.load('./macroeconomic_data/normalized_data.npy', allow_pickle=True).item())
    indicators_labels = dict(np.load('./macroeconomic_data/indicators_labels.npy', allow_pickle=True).item())
    subindicators_labels = dict(np.load('./macroeconomic_data/subindicators_labels.npy', allow_pickle=True).item())

    keys_to_remove = get_incomplete_data_keys(normalized_data, indicators_labels, subindicators_labels)
    filtered_data = filter_dictionary_data(normalized_data, keys_to_remove)


    np.save('./macroeconomic_data/filtered_data.npy', filtered_data)
