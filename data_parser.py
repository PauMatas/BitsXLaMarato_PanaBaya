import pandas as pd
import numpy as np

inpdf = pd.read_csv("COPEDICATClinicSympt_DATA_2020-12-17_1642.csv")
dd = []

useful_columns = [3, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                  29, 30, 31, 32, 34, 35, 36, 39, 43, 45, 47, 49, 53, 55,
                  57, 59, 61, 65, 70, 75, 77, 79, 81, 85, 94, 96, 101, 114,
                  115, 116, 117, 118, 119, 120, 122, 123, 124, 131, 135, 136,
                  138, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                  153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                  165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
                  177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
                  189, 190, 191, 192, 193, 194, 195, 197, 102, 104]

binary_columns = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34,
                  65, 70, 85, 94, 96, 114, 116, 120, 131, 144, 188, 190]

binary_w_uknw_cols = [36, 39, 43, 45, 47, 49, 53, 55, 57, 59, 61, 75, 77, 79,
                      81, 101, 118, 123, 138, 141, 145, 146, 147, 148, 149,
                      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                      161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                      172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
                      183, 184, 185, 186, 102, 104]

date_columns = [117, 122, 189]

others =[6, 7, 16, 17, 18, 35, 115, 119, 124, 135, 143, 187, 191, 192,
         193, 194, 195, 197]

#Corrections
inpdf[118, 624] = indpf[118, 436] = indpf[118, 423] = indpf[118, 713] = indpf[118, 744] = 2
inpdf[118, 764] = indpf[118, 960] = 1
inpdf[123, 696] = indpf[123, 697] = indpf[123, 764] = 2

"""
name [type] (col in df)
"""

def make_up(x, l):
    val = l[x]

    if val == 9 and val in binary_w_uknw_cols:
        return 3

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

    for x in useful_columns:
        patient_data += make_up(x-1, l)

    # sanitary_id [string] (1)
    patient_data.append(l[1])

    return patient_data


def parse():

    dd = [parser_by_row(i, inpdf.iloc[i]) for i in range(len(inpdf))]

    return dd

dd = parse()

headers = ['id'] + [inpdf.columns[i] for i in range(len(inpdf.columns)) if i+1 in useful_columns] + ['_']

pd.DataFrame(dd).to_csv('clean_data.csv', header = headers)

res = [[i for i in range(len(inpdf[135]))], inpdf[135]]
pd.DataFrame(res).to_csv('results_data.csv', header = headers)