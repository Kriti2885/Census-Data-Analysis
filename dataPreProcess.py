"""
@Author : Kriti Upadhyaya

@Order of Execution - 1

@Description:
The program reads the dataset and then find out if there are any missing
values in the dataset. If there are missing values, then the program will
ask for user input. The program implements three ways of fixing the missing value:
1. Removing missing value : All the missing values are removed from the dataset
and a new dataset is created.
2. By replacing the categorical attribute with most frequently occurring value
and continuous attribute with mean of all the values in the dataset.
3. By replacing the categorical attribute with most frequently occurring value
and continuous attribute with median.
The program then writes the updated dataset into a file.
"""

import json


def readData():
    """
    The program reads the data file and creates the dataset. Then the function calls askOption
    function which implements missing value.
    :return: None
    """
    dataSet = []
    with open("adult.data") as datafile:
        for line in datafile:
            line = line.strip()
            record = line.split(", ")
            dataSet.append(record)
    askOption(dataSet)


def askOption(data):
    """
    The function gives option on how to fix the missing values in the dataset.
    User input is captured and corresponding function is called.

    :param data: dataset
    :return: None
    """
    option = input("Select any one of the two options: \n"
                   "1. Remove missing value. \n"
                   "2. Fix missing value using the most prominent occuring categorical "
                   "and mean continuous value. \n"
                   "3. Fix missing value using the most prominent occuring categorical"
                   " and median continuous value. \n")
    if option == 1:
        removeMissingValues(data)
    elif option == 2:
        fixMissingValuesMean(data)
    elif option == 3:
        fixMissingValuesMedian(data)
    else:
        print("Please enter given options!!")
        askOption(data)


def removeMissingValues(data):
    """
    The function reads all the instances from dataset and remove missing value from the dataset.
    The function then calls writeData function to write the updated dataset into a file.

    :param data:
    :return:
    """

    missing = "?"
    finalDataSet = []

    for i in range(len(data)):
        instance = data[i]
        if instance != [''] and missing not in instance:
                finalDataSet.append(instance)
    writeData(finalDataSet)


def fixMissingValuesMean(data):
    """
    The function reads all the instances from dataset and replace missing value from the dataset.
    The categorical attributes are replaced by most frequently occuring vale and the continuous
    attribute is replaced by mean value.
    The function then calls writeData function to write the updated dataset into a file.

    :param data:
    :return:
    """

    missingValue = "?"
    fixedDataSet = []
    for k in range(len(data)):
        instances = data[k]
        if instances != ['']:
            for index in range(14):
                if missingValue in instances[index]:

                    if index == 0: instances[index] = 38
                    elif index == 1: instances[index] = "Private"
                    elif index == 2: instances[index] = 189778
                    elif index == 3: instances[index] = "HS-grad"
                    elif index == 4: instances[index] = 10
                    elif index == 5: instances[index] = "Married-civ-spouse"
                    elif index == 6: instances[index] = "Prof-specialty"
                    elif index == 7: instances[index] = "husband"
                    elif index == 8: instances[index] = "white"
                    elif index == 9: instances[index] = "Male"
                    elif index == 10: instances[index] = 1078
                    elif index == 11: instances[index] = 87
                    elif index == 12: instances[index] = 40
                    elif index == 13: instances[index] = "United-States"
                    else: break
            fixedDataSet.append(instances)

    writeData(fixedDataSet)


def fixMissingValuesMedian(data):
    """
    The function reads all the instances from dataset and replace missing value from the dataset.
    The categorical attributes are replaced by most frequently occuring vale and the continuous
    attribute is replaced by median value.
    The function then calls writeData function to write the updated dataset into a file.

    :param data:
    :return:
    """
    missingValue = "?"
    fixedDataSet = []
    for k in range(len(data)):
        instances = data[k]
        if instances != ['']:
            for index in range(14):
                if missingValue in instances[index]:

                    if index == 0: instances[index] = 37
                    elif index == 1: instances[index] = "Private"
                    elif index == 2: instances[index] = 178356
                    elif index == 3: instances[index] = "HS-grad"
                    elif index == 4: instances[index] = 10
                    elif index == 5: instances[index] = "Married-civ-spouse"
                    elif index == 6: instances[index] = "Prof-specialty"
                    elif index == 7: instances[index] = "husband"
                    elif index == 8: instances[index] = "white"
                    elif index == 9: instances[index] = "Male"
                    elif index == 10: instances[index] = 0
                    elif index == 11: instances[index] = 0
                    elif index == 12: instances[index] = 40
                    elif index == 13: instances[index] = "United-States"
                    else: break

            fixedDataSet.append(instances)

    writeData(fixedDataSet)


def writeData(data):
    """
    Writes the updated dataset into a json file.
    :param data:
    :return:None
    """

    with open('preprocessedData.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    readData()

