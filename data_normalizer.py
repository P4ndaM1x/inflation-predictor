__author__ = '@P4ndaM1x, Michał Rutkowski'

from typing import Callable
from collections import defaultdict
import numpy as np


def modify_subindicators_values(data: dict, modifier: Callable[[int, float], float]) -> dict:
    return {date:
                {indicator_id:
                     {subindicator_id: modifier(indicator_id, value)
                      for subindicator_id, value in subindicators.items()}
                 for indicator_id, subindicators in indicators.items()}
            for date, indicators in data.items()}


def replace_indicator_values(data: dict, replacer: Callable[[int, int, dict], float]) -> dict:
    return {date:
                {indicator_id: replacer(date, indicator_id, subindicators)
                 for indicator_id, subindicators in indicators.items()}
            for date, indicators in data.items()}


def convert_budget_data(data: dict) -> dict:
    monthly_budget_changes = defaultdict(dict)
    previous_values = {0: None, 1: None, 2: None}
    for key, value in sorted({date: indicators[2] for date, indicators in data.items()}.items()):
        for index, month_value in value.items():
            if month_value and previous_values[index]:
                month_change = month_value - previous_values[index]
                monthly_budget_changes[key][index] = month_change
            previous_values[index] = month_value
    return dict(monthly_budget_changes)


parsed_data = dict(np.load('./macroeconomic_data/parsed_data.npy', allow_pickle=True).item())

std_indicator_offset = -100.
std_indicator_modifier = lambda ind_id, val: \
    val if val is None or ind_id in [2, 5] else val + std_indicator_offset
normalized_data = modify_subindicators_values(parsed_data, std_indicator_modifier)

new_budget_indicator = convert_budget_data(normalized_data)
budget_replacer = lambda date, ind_id, val: \
    (new_budget_indicator[date] if date in new_budget_indicator else None) if ind_id == 2 else val
normalized_data = replace_indicator_values(normalized_data, budget_replacer)


np.save('./macroeconomic_data/normalized_data.npy', normalized_data)
