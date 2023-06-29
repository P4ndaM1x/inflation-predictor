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


def filter_labels_inplace(ind_labels: dict, subind_labels: dict, indicators_to_remove: dict[int, list[int]]) -> None:
    for indicator, subindicators in indicators_to_remove.items():
        if not indicators_to_remove[indicator]:
            if indicator in ind_labels.keys():
                del ind_labels[indicator]
            if indicator in subind_labels.keys():
                del subind_labels[indicator]
            continue
        for subind in subindicators:
            if subind in subind_labels[indicator].keys():
                del subind_labels[indicator][subind]
        if not ind_labels[indicator]:
            del ind_labels[indicator]
        if not subind_labels[indicator]:
            del subind_labels[indicator]


def filter_records(data: dict, keys_to_remove: set) -> dict:
    new_data = data.copy()
    for key in keys_to_remove.intersection(set(new_data.keys())):
        del new_data[key]
    return new_data


def filter_indicators(data: dict, indicators_to_remove: dict[int, list[int]]) -> dict:
    new_data = data.copy()
    for key in data.keys():
        for indicator, subindicators in indicators_to_remove.items():
            if not indicators_to_remove[indicator]:
                if indicator in new_data[key].keys():
                    del new_data[key][indicator]
                continue
            for subind in subindicators:
                if subind in new_data[key][indicator].keys():
                    del new_data[key][indicator][subind]
            if not new_data[key][indicator]:
                del new_data[key][indicator]
    return new_data


def filter_data(data: dict, indicators_to_remove: dict[int, list[int]], ind_labels: dict, subind_labels: dict):
    data_filtered_by_indicators = filter_indicators(data, indicators_to_remove)
    keys_to_remove = get_incomplete_data_keys(data_filtered_by_indicators, ind_labels, subind_labels)
    return filter_records(data_filtered_by_indicators, keys_to_remove)


if __name__ == "__main__":
    # usunięcie następujących podwskaźników:
    # 'Wskaźniki cen' -> 'Wskaźniki cen usług transportu i gospodarki magazynowej'
    # 'Wskaźniki cen' -> 'Wskaźniki cen usług telekomunikacji'
    # 'Transport' -> wszystkie
    indicators_to_remove = {0: [30, 33], 8: None}

    normalized_data = dict(np.load('./macroeconomic_data/normalized_data.npy', allow_pickle=True).item())
    normalized_data_recent = dict(np.load('./macroeconomic_data/normalized_data_recent.npy', allow_pickle=True).item())
    indicators_labels = dict(np.load('./macroeconomic_data/indicators_labels.npy', allow_pickle=True).item())
    subindicators_labels = dict(np.load('./macroeconomic_data/subindicators_labels.npy', allow_pickle=True).item())

    filter_labels_inplace(indicators_labels, subindicators_labels, indicators_to_remove)
    filtered_data = filter_data(normalized_data, indicators_to_remove, indicators_labels, subindicators_labels)
    filtered_data_recent = filter_data(normalized_data_recent, indicators_to_remove, indicators_labels,
                                       subindicators_labels)

    np.save('./macroeconomic_data/filtered_data.npy', filtered_data)
    np.save('./macroeconomic_data/filtered_data_recent.npy', filtered_data_recent)
