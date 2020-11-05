import glob
import numpy as np
import pandas as pd
import re


def task1(dataframe: pd.DataFrame, folder_path: str = None):
    """
    Read data from all files to single dataframe using Pandas
    :param folder_path: path to folder with files containing name info
    :param dataframe: Input dataframe with columns: year, name, sex, count
    :return: Output pivot_table dataframe, list of years
    """

    if folder_path is None or folder_path == "":
        files_list = [f for f in glob.glob("*.txt") if "yob" in f]
    else:
        files_list = [f for f in glob.glob(folder_path + "/*.txt") if "yob" in f]

    if len(files_list) == 0:
        print(f"No files found in given directory {folder_path}!")
        return None

    years = []
    for f in files_list[:5]:
        try:
            # Create temporary data frame from file
            temp_dataframe = pd.read_csv(f, header=None, usecols=[0, 1, 2], names=["name", "sex", "count"])
            # Read data's year from file name and add the year to temporary data frame
            year = int(re.findall(r'\d+', f)[0])
            temp_dataframe["year"] = year
            years.append(year)
            # Concatenate existing dataframe with temporary
            dataframe = pd.concat([dataframe, temp_dataframe], ignore_index=True)
        except FileNotFoundError:
            print(f"File {f} doesnt exist!")

    table = pd.pivot_table(dataframe, values="count", index=["year", "name"], columns=["sex"], aggfunc=np.sum)
    return table, years


def task2(dataframe: pd.DataFrame):
    """
    Count unique names
    :param: dataframe: Pandas dataframe with all the necessary data
    :return: Number of unique names
    """

    unique_names = dataframe.groupby('name').nunique()
    return len(unique_names)


def task3(dataframe: pd.DataFrame):
    """
    Count unique names per sex
    :param: dataframe: Pandas dataframe with all the necessary data
    :return:
    number_of_unique_men_names
    number_of_unique_female_names
    """
    unique_names = dataframe.groupby('name').nunique()
    number_of_unique_men_names = unique_names[unique_names["F"] >= 1].count()["F"]
    number_of_unique_female_names = unique_names[unique_names["M"] >= 1].count()["M"]

    return number_of_unique_men_names, number_of_unique_female_names


def task4(dataframe: pd.DataFrame, years):
    """
    Create new columns frequency_male and frequency_female and count a frequency of each name per year
    :param years: list of years
    :param dataframe: dataframe with all necessary data
    :return: dataframe with frequency per sex added
    """

    dataframe["frequency_female"] = 0
    dataframe["frequency_male"] = 0

    for year in years:
        total_births_female_per_year = dataframe.loc[(year, ), "F"].sum()
        total_births_male_per_year = dataframe.loc[(year, ), "M"].sum()
        dataframe.loc[(year,):, "frequency_female"] = dataframe.loc[(year,):, "F"] / total_births_female_per_year
        dataframe.loc[(year,):, "frequency_male"] = dataframe.loc[(year,):, "M"] / total_births_male_per_year

    return dataframe


def main():
    df_names = pd.DataFrame(columns=["year", "name", "sex", "count"])
    # Dataframe with all names and years
    df_names, years = task1(folder_path="names", dataframe=df_names)

    print(f"Number of unique names: {task2(df_names)}")

    number_of_unique_men_names, number_of_unique_female_names = task3(dataframe=df_names)
    print(f"Number of unique men names: {number_of_unique_men_names}")
    print(f"Number of unique female names: {number_of_unique_female_names}")

    df_names = task4(df_names, years)


if __name__ == "__main__":
    main()
