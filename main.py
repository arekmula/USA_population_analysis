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
    print("Starting TASK 1")
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
    return table, dataframe


def task2(dataframe: pd.DataFrame):
    """
    Count unique names

    :param: dataframe: Pandas dataframe with all the necessary data
    :return: Number of unique names
    """
    print("Starting TASK 2")
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
    print("Starting TASK 3")
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
    print("Starting TASK 4")
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

    :param dataframe: dataframe with all necessary data
    :return: year_biggest_ratio: year of the biggest difference between birth of male and female
    :return: year_smallest_ratio: year of the smallest difference between birth of male and female
    """
    print("Starting TASK 5")
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

    :param dataframe: dataframe containing all necessary data,
    :param number_of_top_popular_names: number of top popular names you want to return,
    :return: top_female_names_across_all_years,
    :return: top_male_names_across_all_years,
    """
    print("Starting TASK 6")
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
    Plot changes for Harry, Marilin and top female and male names. On left Y axis plot how many times each name was
    given in a year. On the right Y axis plot popularity of each name

    :param annotate_years: list of years that you want to annotate on a plot
    :param name2: name that you want to plot
    :param name1: name that you want to plot
    :param dataframe: dataframe with all data
    :param top_female_names: sorted series of top female names from which there will be top1 chosen
    :param top_male_names: sorted series of top male names from which there will be top1 chosen
    :return:
    """
    print("Starting TASK 7")
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
        top_names_dataframe_per_year[name] = dataframe.loc[(name,)]["total"]
        top_names_dataframe_per_year[name_freq] = (dataframe.loc[(name,)]["total"] / births_per_year) * 100
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

    # Annotate requested years with corresponding values TODO: Fix arrows
    xytext_positions = [-7, -5, -2]
    for year in annotate_years:
        for name, xytext_position in zip(names, xytext_positions):
            try:
                ax.annotate(f"{name}: {top_names_dataframe_per_year.loc[year, name]}",
                            xy=(year, top_names_dataframe_per_year.loc[year, name]),
                            xytext=(year + xytext_position, top_names_dataframe_per_year.loc[year, name] + 5000),
                            arrowprops=dict(facecolor='black', arrowstyle="-"))
            except KeyError as k:
                print(f"No data about this year: {k}! SKIPPING")

    ax2.set_ylabel("Popularnosc imienia [%]")
    ax2.legend(loc='upper right')


def task8(dataframe: pd.DataFrame, top_female_names: pd.Series, top_male_names: pd.Series):
    """
    Plot sum of popularity of top 1000 names through years. Note the year with biggest difference between male and
     female diversity

    :param dataframe: dataframe with all necessary data
    :param top_female_names: series with most popular female names
    :param top_male_names: series with most popular male names
    :return:
    """
    print("Starting TASK 8")
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
    df_year_changes["Top 1000 F names %"] = (top_female_names_per_year_sum / df_year_changes["F"]) * 100
    df_year_changes["Top 1000 M names %"] = (top_male_names_per_year_sum / df_year_changes["M"]) * 100
    df_year_changes["Difference"] = np.abs(df_year_changes["Top 1000 F names %"] -
                                           df_year_changes["Top 1000 M names %"])
    # Get a year with biggest difference between male and female diversity
    year_biggest_difference_in_diversity = df_year_changes.idxmax()["Difference"]
    print(f"Rok z najwieksza roznica w roznorodnosci miedzy imionami meskimi a zenskimi:"
          f" {year_biggest_difference_in_diversity}")

    fig, ax = plt.subplots(1, 1)
    df_year_changes.plot(y=["Top 1000 F names %", "Top 1000 M names %"], ax=ax)

    fig.suptitle("Udzial 1000 najpopularniejszych imion na przestrzeni lat")
    ax.set_ylabel("Zsumowana popularnosc 1000 najpopularniejszych imion [%]")
    ax.set_xlabel("Rok")
    ax.legend(["Imiona kobiece", "Imiona meskie"])
    ax.annotate(f"Rok z najwieksza roznica \nw roznorodnosci miedzy imionami \nmeskimi a zenskimi:"
                f" {year_biggest_difference_in_diversity}", xy=(year_biggest_difference_in_diversity, ax.get_ylim()[0]),
                xytext=(year_biggest_difference_in_diversity - 30, ax.get_ylim()[0] + 5),
                arrowprops=dict(facecolor='black', arrowstyle="-"))
    ax.grid(axis="both")

    return year_biggest_difference_in_diversity


def task9(dataframe_unpivoted: pd.DataFrame, distinct_years: list, stacked=False, number_of_char_trends_to_plot=3):
    """
    Verify hypothesis that density of last letter in men names changed dramatically.
    - Aggregate data considering year, sex and the last letter
    - Distinct data for 1910, 1960, 2015
    - Normalize data considering sum of births in each year
    - Create bar plot with popularity of each letter and each sex. Take a note which letter had the biggest difference
     between 1910 and 2015
    - For 3 letters with highest difference plot a trend

    :param number_of_char_trends_to_plot: How many characters trends you want to plot
    :param stacked: should bar plot with last character names be stacked or not
    :param distinct_years: Years which you want to consider at bar plot. The first and the last value will be used for
    determining which character had the biggest change across years
    :param dataframe_unpivoted: Dataframe containing all data
    :return:
    """
    print("Starting TASK 9")
    if len(distinct_years) < 2:
        print("You have to give at least 2 years to distinguish!")
        return None

    df_last_characters = pd.DataFrame(columns=["year", "last_character", "sex", "count"])
    df_last_characters[["year", "sex", "count"]] = dataframe_unpivoted[["year", "sex", "count"]]
    # Get last character from each name
    df_last_characters["last_character"] = [name[-1] for name in dataframe_unpivoted["name"].values]

    # Use pivot_table because of duplicate values that has to be summed up.
    # For example (1886, a) occurred multiple times and has to be summed
    df_last_characters = pd.pivot_table(df_last_characters, values="count", index=["year", "last_character"],
                                        columns=["sex"], aggfunc=np.sum)
    df_last_characters = df_last_characters.fillna(0).astype(int)

    # Compute normalized value for each year and sex
    df_last_characters[["F normalized", "M normalized"]] = (df_last_characters[["F", "M"]]
                                                            / df_last_characters.groupby('year').sum())

    # Get dataframe with years that you want to plot
    df_last_characters_distinct_years = pd.DataFrame(df_last_characters.loc[distinct_years])
    # Swap levels so character is on index 0
    df_last_characters_distinct_years = df_last_characters_distinct_years.swaplevel(0, 1)
    # Sort index for faster computing
    df_last_characters_distinct_years = df_last_characters_distinct_years.sort_index()

    fig, ax = plt.subplots(2, 1)

    df_last_characters_distinct_years.groupby("last_character")["F normalized"].head(1000).unstack(level=1).plot.bar(
        stacked=stacked, ax=ax[0])
    df_last_characters_distinct_years.groupby("last_character")["M normalized"].head(1000).unstack(level=1).plot.bar(
        stacked=stacked, ax=ax[1])

    fig.suptitle("Zad9 - Ostatnia litera imienia:")
    ax[0].set_title("W imionach kobiecych")
    ax[0].set_xlabel("Ostatnia litera imienia")
    ax[0].set_ylabel("Popularnosc litery")

    ax[1].set_title("W imionach meskich")
    ax[1].set_xlabel("Ostatnia litera imienia")
    ax[1].set_ylabel("Popularnosc litery")

    # Get list of all last characters from dataframe index
    last_characters = df_last_characters_distinct_years.index.get_level_values(0).unique()
    # Create dataframe with 2 distinguished years which will contain information about difference between those 2 years
    df_year_difference = pd.DataFrame(df_last_characters_distinct_years.loc[(last_characters,
                                                                             (distinct_years[0], distinct_years[-1])),
                                                                            ["F normalized", "M normalized"]])
    # Change sign every 2 rows (in this case for all rows with 1910 year) so you can use .groupby().sum()
    df_year_difference.iloc[::2, :] = df_year_difference.iloc[::2, :] * -1
    # Calculate the difference for all characters between year 1910 and 2015
    df_year_difference = pd.DataFrame(np.abs((df_year_difference.groupby('last_character')).sum()))

    # Find 3 letters with biggest change between year 1910 and 2015 for female and male
    female_chars_biggest_difference = []
    male_chars_biggest_difference = []
    for i in range(number_of_char_trends_to_plot):
        female_char_index = df_year_difference.idxmax()["F normalized"]
        df_year_difference.loc[female_char_index, "F normalized"] = 0  # Set to zero so it's skipped in next iteration
        female_chars_biggest_difference.append(female_char_index)

        male_char_index = df_year_difference.idxmax()["M normalized"]
        df_year_difference.loc[male_char_index, "M normalized"] = 0
        male_chars_biggest_difference.append(male_char_index)

    print(f"Najwiekszą zmianę dla imion kobiecych odnotowano dla liter: {female_chars_biggest_difference}")
    print(f"Najwiekszą zmianę dla imion meskich odnotowano dla liter: {male_chars_biggest_difference}")

    fig, ax = plt.subplots(2, 1)
    fig.suptitle("Zad9 - Popularność ostatniej litery imienia:")
    df_last_characters = df_last_characters.swaplevel(0, 1)
    df_last_characters = df_last_characters.sort_index()

    for female_char in female_chars_biggest_difference:
        df_last_characters.loc[(female_char,), "F normalized"].plot(ax=ax[0])

    ax[0].legend(female_chars_biggest_difference, loc='upper left')
    ax[0].set_title("Wsród kobiet")
    ax[0].set_xlabel("Rok")
    ax[0].set_ylabel("Popularność litery")

    for male_char in male_chars_biggest_difference:
        df_last_characters.loc[(male_char,), "M normalized"].plot(ax=ax[1])

    ax[1].legend(male_chars_biggest_difference, loc='upper left')
    ax[1].set_title("Wsród mężczyzn")
    ax[1].set_xlabel("Rok")
    ax[1].set_ylabel("Popularność litery")


def task10(dataframe: pd.DataFrame):
    """
    Find names that were given both to females and males. Note most popular male and female name.

    :param dataframe: dataframe containing all data
    :return unisex_names: list of names that are both female and male
    :return most_popular_female_unisex_name: most popular female name that is also a male name
    :return most_popular_male_unisex_name: most popular male name that is also a female name
    """
    print("Starting TASK 10")
    # Count in how many years each name existed in each sex. If a name has 0 count in one of the sex column, that means
    # that it never existed as name in this sex and it's not unisex name
    # I could use sum() at the very beginning, but .count() is much faster. So I decided to use count() to get the
    # unisex names first
    df_names_count = (dataframe.groupby('name').count()).astype(int)
    unisex_names = pd.DataFrame(df_names_count.loc[(df_names_count["F"] > 0) & (df_names_count["M"] > 0)]).index.values
    print(f"Imiona unisex: {unisex_names}")

    dataframe = dataframe.swaplevel(0, 1)
    dataframe = dataframe.sort_index()

    df_unisex_names = (dataframe.loc[(unisex_names,),])  # TODO: This is so time consuming
    unisex_names_sum = df_unisex_names.groupby('name').sum()
    most_popular_female_unisex_name = unisex_names_sum.idxmax()["F"]
    most_popular_male_unisex_name = unisex_names_sum.idxmax()["M"]

    return unisex_names, df_unisex_names, most_popular_female_unisex_name, most_popular_male_unisex_name,


def task11(dataframe: pd.DataFrame, df_unisex_names=pd.DataFrame, number_names_to_found=3,
           years_lower_range=(1880, 1920), years_upper_range=(2000, 2020)):
    """
    Find most popular names that were female/male names and then became male/female names
    :param df_unisex_names: dataframe containing information only about unisex names
    :param years_upper_range: lower range of years for which F/M ratio is computed
    :param years_lower_range: upper range of years for which M/F ratio is computed
    :param number_names_to_found: number of names to be found
    :param dataframe: dataframe containing all data
    :return: names with smallest standard deviation between sex and year range, but at the same time with quite big
    number of names given
    """
    print("Starting TASK 11")
    dataframe = dataframe.swaplevel(0, 1)
    dataframe = dataframe.sort_index()

    # Fill NaN with 1 to allow dividing
    df_unisex_names = df_unisex_names.fillna(1)
    # Swap levels so year is the first level
    df_unisex_names = df_unisex_names.swaplevel(0, 1)
    df_unisex_names = df_unisex_names.sort_index()

    # Get list of years created from given range
    lower_years = np.arange(years_lower_range[0], years_lower_range[1])
    upper_years = np.arange(years_upper_range[0], years_upper_range[1])
    years = np.concatenate((lower_years, upper_years))

    # Compute Female to Male ratio and Male to Female ratio in years (1880-1920, 2000-2020)
    df_unisex_names.loc[(years,), "F/M"] = (df_unisex_names.loc[(years,), "F"]
                                            / df_unisex_names.loc[(years,), "M"])
    df_unisex_names.loc[(years,), "M/F"] = (df_unisex_names.loc[(years,), "M"]
                                            / df_unisex_names.loc[(years,), "F"])

    # Choose part of dataframe where F/M ratio is greater than M/F ratio in lower years
    female_names_lower_years = (df_unisex_names[df_unisex_names["F/M"]
                                                > df_unisex_names["M/F"]]).loc[(lower_years,), "F/M"]
    # Choose part of dataframe where F/M ratio is greater than M/F ratio in lower years
    male_names_upper_years = (df_unisex_names[df_unisex_names["F/M"]
                                              < df_unisex_names["M/F"]]).loc[(upper_years,), "M/F"]

    # Sum F/M ratio per name in lower years range and M/F ratio per name in upper years
    df_unisex_names_aggregated = pd.DataFrame(female_names_lower_years.groupby('name').sum())
    df_unisex_names_aggregated["M/F"] = male_names_upper_years.groupby('name').sum()

    # Sort dataframe by F/M ratio to choose strongly female names
    df_unisex_names_sorted_FM = df_unisex_names_aggregated.sort_values(["F/M"], ascending=False)
    # Get index of that sorted names
    df_unisex_names_sorted_FM["F/M index"] = np.arange(0, len(df_unisex_names_sorted_FM))
    # Sort the same dataframe by M/F ratio to choose strongly male names
    df_unisex_names_sorted_MF = df_unisex_names_sorted_FM.sort_values(["M/F"], ascending=False)
    # Get index of that sorted names
    df_unisex_names_sorted_MF["M/F index"] = np.arange(0, len(df_unisex_names_sorted_MF))
    # Calculate sum of that indexes. Then remove NaN
    df_unisex_names_sorted_FM["index sum"] = (df_unisex_names_sorted_FM["F/M index"]
                                              + df_unisex_names_sorted_MF["M/F index"])
    df_unisex_names_sorted_FM = df_unisex_names_sorted_FM[df_unisex_names_sorted_FM["M/F"] > 0]
    # Sort dataframe by index sum in ascending order. The smaller index sum then the more impact the name had in both
    # sex. Choose 20 names with lowest index sum
    df_unisex_names_sorted_FM = df_unisex_names_sorted_FM.sort_values("index sum").head(20)

    # Compute sum between F/M and M/F then sort by highest sum, to distinguish names
    # that had biggest change of popularity between sex.
    # df_unisex_names_sorted_FM["sum"] = df_unisex_names_sorted_FM[["F/M", "M/F"]].sum(axis=1)
    # df_unisex_names_sorted_FM = df_unisex_names_sorted_FM.sort_values("sum", ascending=False)

    # Compute standard deviation between F/M and M/F then sort by smallest std dev, to distinguish names
    # that were almost at the same level of popularity between sex.
    df_unisex_names_sorted_FM["std_dev"] = df_unisex_names_sorted_FM[["F/M", "M/F"]].std(axis=1)
    df_unisex_names_sorted_FM = df_unisex_names_sorted_FM.sort_values("std_dev", ascending=True)
    forgotten_female_names = df_unisex_names_sorted_FM.index.values[:number_names_to_found]
    print(f"{number_names_to_found} Najbardziej zapomniane imiona zenskie, ktore obecnie wystepuja jako meskie:"
          f" {forgotten_female_names}")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    female_plot_styles = ['-b', '-g', '-r']
    male_plot_styles = ['--b', '--g', '--r']
    legends_female = []
    legends_male = []
    for name, female_style, male_style in zip(forgotten_female_names, female_plot_styles, male_plot_styles):
        dataframe.loc[(name,), "F"].plot(ax=ax, style=female_style)
        legends_female.append(f"{name} Female")
        dataframe.loc[(name,), "M"].plot(ax=ax2, style=male_style)
        legends_male.append(f"{name} Male")

    ax.set_title('Przebieg trendu 2 imion, które przez pewien czas były'
                 ' żeńsko/męskie a nastepnie stały się męsko/żeńskie')
    ax.legend(legends_female, loc='upper left')
    ax.grid(axis='x')
    ax.set_ylabel("Liczba nadanych imion żeńskich")
    ax2.set_ylabel("Liczba nadanych imion męskich")
    ax2.legend(legends_male, loc='upper right')
    ax.set_xlabel("Rok")

    return forgotten_female_names


def main():
    df_names = pd.DataFrame(columns=["year", "name", "sex", "count"])
    # Dataframe with all names and years
    df_names, dataframe_no_pivot = task1(folder_path="names", dataframe=df_names)

    print(f"Number of unique names: {task2(df_names)}")

    number_of_unique_men_names, number_of_unique_female_names = task3(dataframe=df_names)
    print(f"Number of unique men names: {number_of_unique_men_names}")
    print(f"Number of unique female names: {number_of_unique_female_names}")

    df_names_freq = task4(df_names)

    year_biggest_ratio, year_smallest_ratio = task5(df_names)
    print(f"Year with biggest difference between birth of female and male: {year_biggest_ratio} and year with the"
          f" smallest difference: {year_smallest_ratio}")

    top_female_names, top_male_names = task6(dataframe=df_names, number_of_top_popular_names=1000)

    task7(dataframe=df_names, top_female_names=top_female_names, top_male_names=top_male_names,
          annotate_years=[1940, 1980, 2019])

    year_biggest_difference_in_diversity = task8(dataframe=df_names, top_female_names=top_female_names,
                                                 top_male_names=top_male_names)

    unisex_names, df_unisex_names, most_popular_female_unisex_name, most_popular_male_unisex_name = task10(df_names)
    print(f"Najpopularniejsze żeńskie imie wystepujace jako męskie: {most_popular_female_unisex_name}.\n"
          f"Najpopularniejsze męskie imie występujące jako żeńskie: {most_popular_male_unisex_name}.")

    forgotten_female_unisex_names = task11(dataframe=df_names, df_unisex_names=df_unisex_names, number_names_to_found=2)

    plt.show()


if __name__ == "__main__":
    main()
