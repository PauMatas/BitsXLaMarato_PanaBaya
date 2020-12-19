import pandas as pd

inpdf = pd.read_csv("COPEDICATClinicSympt_DATA_2020-12-17_1642.csv")
dd = [[]]
useful_columns = [3, 19, 25, 26, 31, 32, 34, 36, 39, 43, 45, 47, 49, 53, 55,
                  57, 59, 61, 65, 70, 75, 77, 79]
binary_columns = [19, 25, 26, 31, 32, 34, 65, 70]
binary_w_uknw_cols = [36, 39, 43, 45, 47, 49, 53, 55, 57, 59, 61, 75, 77, 79]

"""
name [type] (col in df)
"""

def make_up(x):
    val = l[x]

    if val.isnull():
        if val in binary_columns:
            val = 0
        elif val in binary_w_uknw_cols:
            val = 3
        else:
            val = 0

    return val

def parser_by_row(patient_id, l):
    patient_data = []

    # patient_id [int] (0)
    patient_data.append(patient_id)

    patient_data += [make_up(x) for x in useful_columns]

    # sanitary_id [string] (1)
    patient_data.append(l[1])

    dd.append(patient_data)


def parse():
    dd = [parser_by_row(i, inpdf.iloc[i]) for i in range(len(inpdf))]

    print(dd)

parse()
