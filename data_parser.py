import pandas as pd

inpdf = pd.read_csv("COPEDICATClinicSympt_DATA_2020-12-17_1642.csv")
dd = [[]]

"""
name [type] (col in df)
"""
def parser_by_row(patient_id, l):
    patient_data = []

    # patient_id [int] (0)
    patient_data.append(patient_id)

    # sanitary_id [string] (1)
    patient_data.append(l[1])

    dd.append(patient_data)


def parse():
    for i in range(len(inpdf)) : 
        parser_by_row(i, inpdf.iloc[i])

    print(dd)

parse()