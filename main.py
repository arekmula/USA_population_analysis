import glob
import numpy as np
from matplotlib import pyplot as plt
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
    for f in files_list:
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

    table = pd.pivot(dataframe, values="count", index=["year", "name"], columns=["sex"])
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


def task4(dataframe: pd.DataFrame):
    """
    Create new columns frequency_male and frequency_female and count a frequency of each name per year
    :param dataframe: dataframe with all necessary data
    :return: dataframe with frequency per sex added
    """
    birth_per_year_per_sex = dataframe.groupby('year').sum()
    dataframe["frequency_female"] = 0
    dataframe["frequency_male"] = 0
    dataframe[["frequency_female", "frequency_male"]] = dataframe[["F", "M"]]/birth_per_year_per_sex

    return dataframe


def task5(dataframe: pd.DataFrame, years: list):
    """
    Create a plot with 2 subplots, where x is timescale and y is:
    - number of births in year
    - birth of females to birth of males ratio.
    Which year had the biggest and the smallest difference between birth of male and female
    :param years: list of years
    :param dataframe: dataframe with all necessary data
    :return: year_biggest_ratio: year of the biggest difference between birth of male and female
    :return: year_smallest_ratio: year of the smallest difference between birth of male and female
    """

    number_of_births_per_year = []
    number_of_female_births_per_year = []
    number_of_male_births_per_year = []
    female_to_male_birth_ratio_per_year = []
    years.sort()

    for year in years:
        female_births = dataframe.loc[(year, ), "F"].sum()
        number_of_female_births_per_year.append(female_births)

        male_births = dataframe.loc[(year, ), "M"].sum()
        number_of_male_births_per_year.append(male_births)

        female_to_male_birth_ratio_per_year.append(female_births/male_births)

        number_of_births_per_year.append(female_births+male_births)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(years, number_of_births_per_year, '.-')
    ax[0].ticklabel_format(style="plain", axis='y')  # Disable scientific notation
    ax[0].set_ylabel("Liczba narodzin")
    ax[0].set_xlabel("Rok")
    ax[0].set_title('Liczba narodzin na przestrzeni lat w USA')

    x = np.arange(len(years))
    width = 0.3
    # TODO: Ask if bar plot is good or should it be line plot with real ratio?
    ax[1].bar(x - width/2, number_of_female_births_per_year, width, label='Dziewczynki')
    ax[1].bar(x + width/2, number_of_male_births_per_year, width, label='Chlopcy')
    ax[1].set_title('Porównanie liczby narodzin dziewczynek i chłopców na przestrzeni lat w USA')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(years)
    ax[1].tick_params(axis='x', labelrotation=90)
    ax[1].ticklabel_format(style="plain", axis='y')
    ax[1].set_ylabel("Liczba narodzin")
    ax[1].set_xlabel("Rok")
    ax[1].legend(loc='upper left')

    # Add secondary y axis
    ax2 = ax[1].twinx()
    ax2.plot(x, female_to_male_birth_ratio_per_year, '.-r')
    ax2.set_ylabel("Stosunek narodzin dziewczynek do chlopcow", color='red')

    # Calculate in which year the ratio was biggest:
    biggest_index = np.argmax(np.abs(np.ones(len(years)) - female_to_male_birth_ratio_per_year))
    smallest_index = np.argmin(np.abs(np.ones(len(years)) - female_to_male_birth_ratio_per_year))
    return years[int(biggest_index)], years[int(smallest_index)]


def main():
    df_names = pd.DataFrame(columns=["year", "name", "sex", "count"])
    # Dataframe with all names and years
    df_names, years = task1(folder_path="names", dataframe=df_names)

    print(f"Number of unique names: {task2(df_names)}")

    number_of_unique_men_names, number_of_unique_female_names = task3(dataframe=df_names)
    print(f"Number of unique men names: {number_of_unique_men_names}")
    print(f"Number of unique female names: {number_of_unique_female_names}")

    df_names = task4(df_names)
    year_biggest_ratio, year_smallest_ratio = task5(df_names, years)
    print(f"Year with biggest difference between birth of female and male: {year_biggest_ratio} and year with the"
          f" smallest difference: {year_smallest_ratio}")

    plt.show()


if __name__ == "__main__":
    main()
