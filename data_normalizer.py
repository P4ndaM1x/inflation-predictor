__author__ = '@P4ndaM1x, Michał Rutkowski'

from typing import Callable
from collections import defaultdict
import numpy as np
import functools


def modify_subindicators_values(data: dict, modifier: Callable[[int, int, float], float]) -> dict:
    return {date:
            {indicator_id:
             {subindicator_id: modifier(indicator_id, subindicator_id, value)
              for subindicator_id, value in subindicators.items()}
             for indicator_id, subindicators in indicators.items()}
            for date, indicators in data.items()}


def replace_indicator_values(data: dict, replacer: Callable[[int, int, dict], float]) -> dict:
    return {date:
            {indicator_id: replacer(date, indicator_id, subindicators)
             for indicator_id, subindicators in indicators.items()}
            for date, indicators in data.items()}


def convert_budget_data(data: dict) -> dict:
    indicators_labels = dict(np.load('./macroeconomic_data/indicators_labels.npy', allow_pickle=True).item())
    budget_key = None
    for key, val in indicators_labels.items():
        if val == 'Budżet państwa':
            budget_key = key
            break
    if budget_key is None:
        return None
    monthly_budget_changes = defaultdict(dict)
    previous_values = {0: 0, 1: 0, 2: 0}
    for key, value in sorted({date: indicators[budget_key] for date, indicators in data.items()}.items()):
        monthly_budget_changes[key] = {0: None, 1: None, 2: None}
        for index, month_value in value.items():
            if month_value:
                month_change = month_value - previous_values[index]
                monthly_budget_changes[key][index] = month_change
            previous_values[index] = month_value
    return dict(monthly_budget_changes)


def scale_data_with_minmax(data: dict, minmax_values: dict) -> dict:
    def scaler(val: float, minmax: tuple):
        return (val - minmax[0]) / (minmax[1] - minmax[0]) * 2 - 1

    def minmax_modifier(ind_id: int, subind_id: int, val: float):
        return val if val is None else scaler(val, minmax_values[(ind_id, subind_id)])

    return modify_subindicators_values(data, minmax_modifier)


def unscale_data_with_minmax(data: np.ndarray, *, ind_id: int = 0, subind_id: int = 36):
    minv, maxv = dict(
        np.load('./macroeconomic_data/minmax_values.npy', allow_pickle=True).item()
    )[(ind_id, subind_id)]
    std_indicator_offset = 100. if ind_id not in [2, 5] else 0
    return (data + 1) * (maxv - minv) / 2.0 + minv + std_indicator_offset


if __name__ == "__main__":
    parsed_data = dict(np.load('./macroeconomic_data/parsed_data.npy', allow_pickle=True).item())

    std_indicator_offset = -100.
    std_indicator_modifier = lambda ind_id, subind_id, val: \
        val if val is None or ind_id in [2, 5] else val + std_indicator_offset
    normalized_data = modify_subindicators_values(parsed_data, std_indicator_modifier)

    new_budget_indicator = convert_budget_data(normalized_data)
    if new_budget_indicator is not None:
        budget_replacer = lambda date, ind_id, val: \
            (new_budget_indicator[date] if date in new_budget_indicator else None) if ind_id == 2 else val
        normalized_data = replace_indicator_values(normalized_data, budget_replacer)

    def reducing_func(minmax_values: dict, value_tuple: tuple):
        key, value = value_tuple
        minv, maxv = minmax_values.setdefault(key, (1e5, -1e5))
        minmax_values[key] = (min(minv, value), max(maxv, value))
        return minmax_values

    data_as_list = [((ind_id, subind_id), value)
                    for date, indicators in normalized_data.items()
                    for ind_id, subindicators in indicators.items()
                    if subindicators is not None
                    for subind_id, value in subindicators.items()
                    if value is not None]

    minmax_values = functools.reduce(reducing_func, data_as_list, dict())
    normalized_data = scale_data_with_minmax(normalized_data, minmax_values)


    np.save('./macroeconomic_data/normalized_data.npy', normalized_data)
    np.save('./macroeconomic_data/minmax_values.npy', minmax_values)
