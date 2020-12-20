import pandas as pd
import numpy as np
import sys

def clean_data(file_name):
  try:
    aa = pd.read_csv(file_name)
  except:
    print("File: '" + file_name + "' does not exist.")
    sys.exit()

  dd = []
  inpdf = aa[aa['final_diagnosis_code'] != 2]
  inpdf = inpdf.replace([np.inf, -np.inf], np.nan)
  inpdf = inpdf.dropna(subset = ['final_diagnosis_code'])

  useful_columns = [3, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 34, 35, 36, 39, 43, 45, 47, 49, 53, 55,
                    57, 59, 61, 65, 69, 70, 73, 75, 77, 79, 81, 85, 87, 94, 95, 96, 97, 101, 102, 108, 104, 114,
                    115, 131, 141, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                    153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
                    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
                    177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 188, 58,
                    189, 190, 191, 192, 193, 194, 197]

  binary_columns = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34,
                    65, 70, 85, 94, 96, 114, 116, 120, 131, 144, 188, 190, 97, 95, 58, 73, 87]

  binary_w_uknw_cols = [36, 39, 43, 45, 47, 49, 53, 55, 57, 59, 61, 75, 77, 79,
                        81, 101, 108, 118, 123, 141, 145, 146, 147, 148, 149,
                        150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
                        161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                        172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
                        183, 184, 185, 186, 102, 104, 69]

  date_columns = [117, 122, 189, 194]

  others =[6, 7, 16, 17, 18, 35, 115, 119, 124, 143, 187, 191, 192,
           193, 194, 195, 197]
  #Corrections
  inpdf[118, 624] = inpdf[118, 436] = inpdf[118, 423] = inpdf[118, 713] = inpdf[118, 744] = 2
  inpdf[118, 764] = inpdf[118, 960] = 1
  inpdf[123, 696] = inpdf[123, 697] = inpdf[123, 764] = 2

  def make_up(x, l):
      val = l[x]

      if val == 9 and x in binary_w_uknw_cols:
          return 3

      if isinstance(val, str) or np.isnan(val):
          if x == 17:
              return 4 # Unknown survey type
          if x in binary_columns:
              return 0
          elif x in binary_w_uknw_cols or x == 18:
              return 3
          else:
              return 0

      return val

  def parser_by_row(patient_id, l):
      patient_data = [make_up(x-1, l) for x in useful_columns]
      return patient_data

  dd = [parser_by_row(i, inpdf.iloc[i]) for i in range(len(inpdf))]


  headers = [inpdf.columns[i] for i in range(len(inpdf.columns)) if i+1 in useful_columns]

  dd = pd.DataFrame(dd)
  dd.to_csv('clean_data.csv', header = headers, index = False)

  res = np.array(inpdf.iloc[:,135])

  res = pd.DataFrame(res)
  res.to_csv('diagnosis_data.csv', header = ['final_diagnosis_code'], index = False)

  if(not np.any(np.isnan(dd)) and np.all(np.isfinite(dd)) and not np.any(np.isnan(res)) and np.all(np.isfinite(res))):
        print("Data cleaning done.")
  else:
        print("Error in values.")