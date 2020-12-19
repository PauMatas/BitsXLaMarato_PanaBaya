import pandas as pd
import numpy as np

inpdf = pd.read_csv("COPEDICATClinicSympt_DATA_2020-12-17_1642.csv")
dd = []
useful_columns = [3, 19, 25, 26, 31, 32, 34, 36, 39, 43, 45, 47, 49, 53, 55,
                  57, 59, 61, 65, 70, 75, 77, 79]
binary_columns = [19, 25, 26, 31, 32, 34, 65, 70]
binary_w_uknw_cols = [36, 39, 43, 45, 47, 49, 53, 55, 57, 59, 61, 75, 77, 79]

"""
name [type] (col in df)
"""

def make_up(x, l):
    val = l[x]

    if isinstance(val, str) or np.isnan(val):
        if val in binary_columns:
            return 0
        elif val in binary_w_uknw_cols:
            return 3
        else:
            return 0

    return val

def parser_by_row(patient_id, l):
    patient_data = []

    # patient_id [int] (0)
    patient_data.append(patient_id)

    patient_data += [make_up(x-1, l) for x in useful_columns]

    # sanitary_id [string] (1)
    patient_data.append(l[1])

    return patient_data


def parse():

    dd = [parser_by_row(i, inpdf.iloc[i]) for i in range(len(inpdf))]

    return dd

dd = parse()

headers = ['id'] + [inpdf.columns[i] for i in range(len(inpdf.columns)) if i+1 in useful_columns] + ['_']

pd.DataFrame(dd).to_csv('clean_data.csv', header = headers)
