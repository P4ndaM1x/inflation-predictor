import pandas as pd
import numpy as np

rawdata_path = './macroeconomic_data/'
rawdata_filename_suffix = '_miesięczne.csv'
indicators_labels = {
    0: 'Wskaźniki cen',  # dane od 2000 roku
    1: 'Budownictwo',  # dane od 2000 roku
    2: 'Budżet państwa',  # dane od 2000 roku (od początku roku do końca okresu w mln zł!)
    3: 'Handel wewnętrzny',  # dane od 2006 roku
    4: 'Handel zagraniczny',  # dane od 2000 roku
    5: 'Koniunktura konsumencka i gospodarcza',  # dane od 2000 roku
    6: 'Przemysł',  # dane od 2005 roku
    7: 'Rynek pracy',  # dane od 2000 roku
    8: 'Transport',  # dane od 2011 roku
    9: 'Wynagrodzenia i świadczenia społeczne'  # dane od 2000 roku
}

def parse_with_years_range(years_range):
    indicators_list = []
    subindicators_labels = {}
    for indicator_id, indicator_name in indicators_labels.items():

        filename = rawdata_path + indicator_name + rawdata_filename_suffix
        df = pd.read_csv(filename, sep=';', header=[0, 1])
        if indicator_id in [2, 5]:
            interesting_rows = df[df['Jednostka']['Unnamed: 2_level_1'] != 'pusta']
        else:
            interesting_rows = df[df['Jednostka']['Unnamed: 2_level_1'] == 'okres poprzedni=100']

        labels = list(interesting_rows.iloc[:, [0, 1]].to_dict().values())
        subindicators_labels[indicator_id] = {date: ('; '.join([str(d[date]) for d in labels]).replace('; nan', ''))
                                              for date in labels[0]}

        interesting_columns = [column for column in df.columns
                               if column[0].isdigit() and years_range[0] <= int(column[0]) <= years_range[1]]
        data = interesting_rows[interesting_columns].to_dict()
        data = {tuple(map(int, k)):
                    {indicator_id: {key: None if value == '.' else float(str(value).replace(',', '.'))
                                    for key, value in d.items()}}
                for k, d in data.items()}
        indicators_list.append(data)

    parsed_data = {date: {k: v for d in indicators_list if date in d for k, v in d[date].items()}
                   for date in indicators_list[0]}

    return (parsed_data, subindicators_labels)


parsed_data, subindicators_labels = parse_with_years_range((2000, 2022))

# format: {int(indicator_id): str(label), ...}
np.save(rawdata_path + 'indicators_labels.npy', indicators_labels)
# format: {int(indicator_id): {int(subindicator_id): str(label), ...}, ...}
np.save(rawdata_path + 'subindicators_labels.npy', subindicators_labels)
# format: {(int(year), int(month)): {int(indicator_id): {int(subindicator_id): float(value), ...}, ...}, ...}
# cells marked with a dot in `.csv` files are represented by None values
np.save(rawdata_path + 'parsed_data.npy', parsed_data)

parsed_data_recent, subindicators_labels_recent = parse_with_years_range((2023, 2023))
assert subindicators_labels == subindicators_labels_recent, "Subindicators differ between years"

# format same as parsed_data
np.save(rawdata_path + 'parsed_data_recent.npy', parsed_data_recent)
