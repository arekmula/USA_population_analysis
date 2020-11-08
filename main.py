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

    for f in files_list:
        try:
            # Create temporary data frame from file
            temp_dataframe = pd.read_csv(f, header=None, usecols=[0, 1, 2], names=["name", "sex", "count"])
            # Read data's year from file name and add the year to temporary data frame
            temp_dataframe["year"] = int(re.findall(r'\d+', f)[0])
            # Concatenate existing dataframe with temporary
            dataframe = pd.concat([dataframe, temp_dataframe], ignore_index=True)
        except FileNotFoundError:
            print(f"File {f} doesnt exist!")

    table = pd.pivot(dataframe, values="count", index=["year", "name"], columns=["sex"])
    return table


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
    number_of_unique_men_names = unique_names[unique_names["M"] >= 1].count()["M"]
    number_of_unique_female_names = unique_names[unique_names["F"] >= 1].count()["F"]

    return number_of_unique_men_names, number_of_unique_female_names


def task4(dataframe: pd.DataFrame):
    """
    Create new columns frequency_male and frequency_female and count a frequency of each name per year
    :param dataframe: dataframe with all necessary data
    :return: dataframe with frequency per sex added
    """
    # Calculate sum of births per year and sex
    birth_per_year_per_sex = dataframe.groupby('year').sum()
    dataframe["frequency_female"] = 0
    dataframe["frequency_male"] = 0
    dataframe[["frequency_female", "frequency_male"]] = dataframe[["F", "M"]] / birth_per_year_per_sex

    return dataframe


def task5(dataframe: pd.DataFrame):
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

    # Calculate sum of births per year and sex
    birth_per_year_per_sex = dataframe[["F", "M"]].groupby('year').sum()
    birth_per_year_per_sex["total"] = birth_per_year_per_sex["F"] + birth_per_year_per_sex["M"]
    birth_per_year_per_sex["Female to Male birth ratio"] = birth_per_year_per_sex["F"] / birth_per_year_per_sex["M"]

    fig, ax = plt.subplots(2, 1)

    birth_per_year_per_sex.plot(y=["total"], ax=ax[0])
    ax[0].ticklabel_format(style="plain", axis='y')  # Disable scientific notation
    ax[0].set_ylabel("Liczba narodzin")
    ax[0].set_xlabel("Rok")
    ax[0].set_title('Liczba narodzin na przestrzeni lat w USA')
    ax[0].get_legend().remove()
    ax[0].minorticks_on()
    ax[0].grid(axis="x")

    birth_per_year_per_sex["Reference"] = 1  # Create column with value 1, to better visualization of ratio on plot
    birth_per_year_per_sex.plot(y=["Female to Male birth ratio"], ax=ax[1])
    birth_per_year_per_sex.plot(y=["Reference"], ax=ax[1], style='--r')
    ax[1].set_title('Stosunek liczby narodzin dziewczynek do chłopców na przestrzeni lat w USA')
    ax[1].set_ylabel("Stosunek liczby narodzin dziewczynek do chłopców")
    ax[1].set_xlabel("Rok")
    ax[1].legend(["Female to Male birth ratio", "Reference"])
    ax[1].minorticks_on()
    ax[1].grid(axis="x")

    # Calculate in which year the ratio was biggest:
    biggest_ratio_index = np.argmax(np.abs(np.ones(len(birth_per_year_per_sex["Female to Male birth ratio"]))
                                           - birth_per_year_per_sex["Female to Male birth ratio"]))
    smallest_ratio_index = np.argmin(np.abs(np.ones(len(birth_per_year_per_sex["Female to Male birth ratio"]))
                                            - birth_per_year_per_sex["Female to Male birth ratio"]))

    ax[1].annotate(f"Najwieksza roznica: {birth_per_year_per_sex.index[biggest_ratio_index]}",
                   (birth_per_year_per_sex.index[biggest_ratio_index],
                    birth_per_year_per_sex.iloc[biggest_ratio_index, 3]),
                   arrowprops=dict(facecolor='black', arrowstyle="-"))
    ax[1].annotate(f"Najmniejsza roznica: {birth_per_year_per_sex.index[smallest_ratio_index]}",
                   (birth_per_year_per_sex.index[smallest_ratio_index],
                    birth_per_year_per_sex.iloc[smallest_ratio_index, 3]),
                   arrowprops=dict(facecolor='black', arrowstyle="-"))

    return birth_per_year_per_sex.index[biggest_ratio_index], birth_per_year_per_sex.index[smallest_ratio_index]


def task6(dataframe: pd.DataFrame, number_of_top_popular_names: int):
    """
    Get 1000 most popular names for each sex in. The method should get 1000 most popular names for each year and sex and
    then sum them up to get 1000 most popular names for each sex.
    :param dataframe - dataframe containing all necessary data,
    :param number_of_top_popular_names - number of top popular names you want to return,
    :return: top_female_names_across_all_years,
    :return: top_male_names_across_all_years,
    """
    # First sort by column values and then sort by index value
    female_names_sorted = (dataframe.sort_values('F', ascending=False)).sort_index(level=[0], ascending=[True])["F"]
    male_names_sorted = (dataframe.sort_values('M', ascending=False)).sort_index(level=[0], ascending=[True])["M"]

    # Get popular names per year
    female_names_sorted = female_names_sorted.groupby('year').head(number_of_top_popular_names)
    male_names_sorted = male_names_sorted.groupby('year').head(number_of_top_popular_names)

    # Get the most popular names across all years
    top_female_names_across_all_years = (female_names_sorted.groupby('name').sum()).sort_values(ascending=False)
    top_female_names_across_all_years = top_female_names_across_all_years.head(number_of_top_popular_names)
    top_male_names_across_all_years = (male_names_sorted.groupby('name').sum()).sort_values(ascending=False)
    top_male_names_across_all_years = top_male_names_across_all_years.head(number_of_top_popular_names)

    return top_female_names_across_all_years, top_male_names_across_all_years


def task7(dataframe: pd.DataFrame, top_female_names: pd.Series, top_male_names: pd.Series, annotate_years: list,
          name1: str = "Harry", name2: str = "Marilin"):
    """
    Plot changes for Harry, Marilin and top female and male names. On left Y axis plot how many times each name was given
    in a year. On the right Y axis plot popularity of each name
    :param annotate_years: list of years that you want to annotate on a plot
    :param name2: name that you want to plot
    :param name1: name that you want to plot
    :param dataframe: dataframe with all data
    :param top_female_names: sorted series of top female names from which there will be top1 chosen
    :param top_male_names: sorted series of top male names from which there will be top1 chosen
    :return:
    """

    top_female_name = top_female_names.index[0]
    top_male_name = top_male_names.index[0]

    dataframe = dataframe.fillna(0)
    dataframe["total"] = dataframe["F"] + dataframe["M"]

    names = [top_male_name, top_female_name, name1, name2]
    names_popularity = [top_male_name + " popularity", top_female_name + " popularity", name1 + " popularity",
                        name2 + " popularity"]
    top_names_dataframe_per_year = pd.DataFrame(columns=names)
    top_names_dataframe_per_year = pd.concat([top_names_dataframe_per_year, pd.DataFrame(columns=names_popularity)])

    births_per_year = dataframe["total"].groupby("year").sum()

    dataframe = dataframe.swaplevel(0, 1)
    # Sort index for faster computing
    # https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
    dataframe = dataframe.sort_index()
    for name, name_freq in zip(names, names_popularity):
        top_names_dataframe_per_year[name] = dataframe.loc[(name, )]["total"]
        top_names_dataframe_per_year[name_freq] = (dataframe.loc[(name, )]["total"] / births_per_year) * 100
    top_names_dataframe_per_year = top_names_dataframe_per_year.fillna(0)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Zadanie 7")
    # Add secondary y axis
    ax2 = ax.twinx()
    top_names_dataframe_per_year.plot(y=names, ax=ax)
    top_names_dataframe_per_year.plot(y=names_popularity, ax=ax2, style='--')

    ax.set_ylabel("Liczba nadanych imion")
    ax.set_xlabel("Rok")
    ax.legend(loc='upper left')

    # Annotate requested years with corresponding values
    for year in annotate_years:
        for name in names:
            try:
                ax.annotate(f"{name}: {top_names_dataframe_per_year.loc[year, name]}",
                            (year, top_names_dataframe_per_year.loc[year, name]),
                            arrowprops=dict(facecolor='black', arrowstyle="-"))
            except KeyError as k:
                print(f"No data about this year: {k}! SKIPPING")

    ax2.set_ylabel("Popularnosc imienia [%]")
    ax2.legend(loc='upper right')


def task8(dataframe: pd.DataFrame, top_female_names: pd.Series, top_male_names: pd.Series):
    """
    Plot sum of popularity of top 1000 names through years.
    :param dataframe: dataframe with all necessary data
    :param top_female_names: series with most popular female names
    :param top_male_names: series with most popular male names
    :return:
    """

    dataframe = dataframe.fillna(0)
    # Sum female and male names per year
    df_year_changes = dataframe.groupby('year').sum()
    df_year_changes["Top 1000 F names %"] = 0
    df_year_changes["Top 1000 M names %"] = 0

    # Swap year with data for better iterating
    dataframe = dataframe.swaplevel(0, 1)
    # Sort index for faster computing
    dataframe = dataframe.sort_index()

    # Find all top 1000 names and sum them by years
    top_female_names_per_year_sum = dataframe.loc[top_female_names.index].groupby("year").sum()["F"]
    top_male_names_per_year_sum = dataframe.loc[top_male_names.index].groupby("year").sum()["M"]

    # Calculate popularity
    df_year_changes["Top 1000 F names %"] = (top_female_names_per_year_sum/df_year_changes["F"]) * 100
    df_year_changes["Top 1000 M names %"] = (top_male_names_per_year_sum/df_year_changes["M"]) * 100

    fig, ax = plt.subplots(1, 1)
    df_year_changes.plot(y=["Top 1000 F names %", "Top 1000 M names %"], ax=ax)

    fig.suptitle("Udzial 1000 najpopularniejszych imion na przestrzeni lat")
    ax.set_ylabel("Zsumowana popularnosc 1000 najpopularniejszych imion [%]")
    ax.set_xlabel("Rok")
    ax.legend(["Imiona kobiece", "Imiona meskie"])
    ax.grid(axis="both")


def main():
    df_names = pd.DataFrame(columns=["year", "name", "sex", "count"])
    # Dataframe with all names and years
    df_names = task1(folder_path="names", dataframe=df_names)

    print(f"Number of unique names: {task2(df_names)}")

    number_of_unique_men_names, number_of_unique_female_names = task3(dataframe=df_names)
    print(f"Number of unique men names: {number_of_unique_men_names}")
    print(f"Number of unique female names: {number_of_unique_female_names}")

    df_names = task4(df_names)

    year_biggest_ratio, year_smallest_ratio = task5(df_names)
    print(f"Year with biggest difference between birth of female and male: {year_biggest_ratio} and year with the"
          f" smallest difference: {year_smallest_ratio}")

    top_female_names, top_male_names = task6(dataframe=df_names, number_of_top_popular_names=1000)

    task7(dataframe=df_names, top_female_names=top_female_names, top_male_names=top_male_names,
          annotate_years=[1940, 1980, 2019])

    task8(dataframe=df_names, top_female_names=top_female_names, top_male_names=top_male_names)
    plt.show()


if __name__ == "__main__":
    main()
