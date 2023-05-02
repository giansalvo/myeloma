import pandas as pd
import csv          # importing the csv module

PATH_ORIG = 'myeloma-orig.csv'
PATH_WORK = 'myeloma-work.csv'
FIELD_SEPARATOR = ';'

FIELD_YEAR_OF_DIAGNOSIS = "Year of diagnosis (1975-2019 by 5)"
FIELD_PATIENT_ID = "Patient ID"
FIELD_YEAR_FOLLOW_UP = "Year of follow-up recode"
FIELD_YEAR_DEATH = "Year of death recode"
FIELD_AGE = "Age recode with single ages and 90+"
FIELD_SEQUENCE = "Sequence number"
FIELD_SEER_CAUSE_SPECIFIC = "SEER cause-specific death classification"
FIELD_SEER_OTHER_CAUSE = "SEER other cause of death classification"
FIELD_SURVIVAL = "Survival months"
FIELD_COD_TO_SITE = "COD to site recode"
FIELD_VITAL_STATUS = "Vital status recode (study cutoff used)"
FIELD_RADIATION = "Radiation recode"
FIELD_CHEMOTHERAPY = "Chemotherapy recode (yes, no/unk)"
FIELD_MONTHS_FROM_DIAG_TO_TREAT = "Months from diagnosis to treatment"
FIELD_PRIMARY_SITE = "Primary Site - labeled"
FIELD_HIST_BEHAV = "ICD-O-3 Hist/behav, malignant"

def write_header(f):
    print("toto", file=f)


def main():
    # field names
    fields = ['a', 'b', 'c', 'd']
    mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
              {'a': 100, 'b': 200, 'c': 300, 'd': 400},
              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]

    mydict2 = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
              {'a': 100, 'b': 200, 'c': 300, 'd': 400},
              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]


    df = pd.DataFrame(mydict)
    print(df)
    print(df.iloc[1])
    print(df.iloc[1]['c'])

    # writing to csv file
    with open(PATH_WORK, 'w', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, delimiter=FIELD_SEPARATOR, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()


        # writing data rows
        writer.writerows(mydict2)
        writer.writerows(mydict2)


    ############

    df = pd.read_csv(PATH_WORK, sep=FIELD_SEPARATOR)
    print(df)
    print(df.info())
    print(df.isnull().sum())
    i = 0
    for row in df.iterrows():
        print(df.iloc[i]['c'])
        i += 1

if __name__ == '__main__':
    main()

