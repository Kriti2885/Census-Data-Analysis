"""
@Author : Kriti Upadhyaya

@Order of Execution -2

@Description:
The program implements naive Bayes algorithm for classification.
Naive Bayes Algorithm is :
P(H|X) = P(X|H)*P(H)/ P(X)
To handle continuous variable, equal width binning and probability distribution
function is implemented.

"""

from __future__ import division
import time
import json
import math
from statistics import mean, stdev, variance
from collections import defaultdict, Counter


def readData():
    """
    The function reads json data file created in program 2 in which the missing values
    have been handled and the continuous attributes have been discretized.

    :return: dataSet
    """

    dataSet =[]
    with open('preprocessedData.json') as f:
        data = json.load(f)
    for line in data:
        dataSet.append(line)
    return dataSet


def askOption():
    """
    The function gives user option to select how to discretize the continuous values.
    Depending on the user selection, calls the respective function.

    :param data: dataset
    :return: updated dataset
    """

    option = input("Select any one of the two approaches: \n"
                   "1. Equal Width Binning \n"
                   "2. Probability Distribution \n")

    return option


def equalWidthBinning(data):
    """
    In this function, continuous attributes are divided into bins of equal width and then
    the bins are allotted a discrete value and the continuous values are converted to discrete.

    :param data: dataset
    :return: updated dataset
    """

    for i in range(0, len(data)):
        record = data[i]

        # age
        if 0 <= int(record[0]) <= 10: record[0] = 0
        elif 11 <= int(record[0]) <= 20: record[0] = 1
        elif 21 <= int(record[0]) <= 30: record[0] = 2
        if 31 <= int(record[0]) <= 40: record[0] = 3
        if 41 <= int(record[0]) <= 50: record[0] = 4
        if 51 <= int(record[0]) <= 60: record[0] = 5
        elif 61 <= int(record[0]) <= 70: record[0] = 6
        elif 71 <= int(record[0]) <= 80: record[0] = 7
        elif 81 <= int(record[0]) <= 90: record[0] = 8
        else: record[0] = 9

        # finalweight 150k, 23k
        if 0 <= int(record[2]) < 150000: record[2] = 0
        elif 150000 <= int(record[2]) < 300000: record[2] = 1
        elif 300000 <= int(record[2]) < 450000: record[2] = 2
        elif 450000 <= int(record[2]) < 600000: record[2] = 3
        elif 600000 <= int(record[2]) < 750000: record[2] = 4
        elif 750000 <= int(record[2]) < 900000: record[2] = 5
        elif 900000 <= int(record[2]) < 1050000: record[2] = 6
        elif 1050000 <= int(record[2]) < 1200000: record[2] = 7
        elif 1200000 <= int(record[2]) < 1350000: record[2] = 8
        elif 1350000 <= int(record[2]) < 1500000: record[2] = 9

        # education_num
        if 0 <= int(record[4]) < 3: record[4] = 0
        elif 3 <= int(record[4]) < 6: record[4] = 1
        elif 6 <= int(record[4]) < 9: record[4] = 2
        elif 9 <= int(record[4]) < 12: record[4] = 3
        elif 12 <= int(record[4]) < 15: record[4] = 4
        elif 15 <= int(record[4]) < 18: record[4] = 5

        # Capital gain
        if 0 <= int(record[10]) < 10000: record[10] = 0
        elif 10000 <= int(record[10]) < 20000: record[10] = 1
        elif 20000 <= int(record[10]) < 30000: record[10] = 2
        elif 30000 <= int(record[10]) < 40000: record[10] = 3
        elif 40000 <= int(record[10]) < 50000: record[10] = 4
        elif 50000 <= int(record[10]) < 60000: record[10] = 5
        elif 60000 <= int(record[10]) < 70000: record[10] = 6
        elif 70000 <= int(record[10]) < 80000: record[10] = 7
        elif 80000 <= int(record[10]) < 90000: record[10] = 8
        elif 90000 <= int(record[10]) < 100000: record[10] = 9


        # capitalLoss
        if 0 <= int(record[11]) < 500: record[11] = 0
        elif 500 <= int(record[11]) < 1000: record[11] = 1
        elif 1000 <= int(record[11]) < 1500: record[11] = 2
        elif 1500 <= int(record[11]) < 2000: record[11] = 3
        elif 2000 <= int(record[11]) < 2500: record[11] = 4
        elif 2500 <= int(record[11]) < 3000: record[11] = 5
        elif 3000 <= int(record[11]) < 3500: record[11] = 6
        elif 3500 <= int(record[11]) < 4000: record[11] = 7
        elif 4000 <= int(record[11]) < 4500: record[11] = 8


        # hours per week
        if 0 <= int(record[12]) < 6: record[12] = 0
        elif 6 <= int(record[12]) < 11: record[12] = 1
        elif 11 <= int(record[12]) < 16: record[12] = 2
        elif 16 <= int(record[12]) < 21: record[12] = 3
        elif 21 <= int(record[12]) < 26: record[12] = 4
        elif 26 <= int(record[12]) < 31: record[12] = 5
        elif 31 <= int(record[12]) < 36: record[12] = 6
        elif 36 <= int(record[12]) < 41: record[12] = 7
        elif 41 <= int(record[12]) < 46: record[12] = 8
        elif 46 <= int(record[12]) < 51: record[12] = 9
        elif 51 <= int(record[12]) < 56: record[12] = 10
        elif 56 <= int(record[12]) < 61: record[12] = 11
        elif 61 <= int(record[12]) < 66: record[12] = 12
        elif 66 <= int(record[12]) < 71: record[12] = 13
        elif 71 <= int(record[12]) < 76: record[12] = 14
        elif 76 <= int(record[12]) < 81: record[12] = 15
        elif 81 <= int(record[12]) < 86: record[12] = 16
        elif 86 <= int(record[12]) < 91: record[12] = 17
        elif 91 <= int(record[12]) < 96: record[12] = 18
        elif 96 <= int(record[12]) < 111: record[12] = 19
    return data


def kFoldCrossValidation(dataSet):
    """
    The function divides dataset into 10 folds of equal size and returns them to the main function.

    :param dataSet: dataset with test and train.
    :return: folds
    """

    length = int(len(dataSet)/10)
    folds = []
    for i in range(9):
        folds += [dataSet[i*length:(i+1)*length]]
    folds += [dataSet[9*length: len(dataSet)]]
    return folds


def dataCleanup(data):
    """
    The function creates a dictionary with classes as keys and values as the instances
    belonging to that class. The input is train fold dataset.

    :param data: train fold
    :return: classDistribution : dictionary with classes as keys and instances as values.
    """

    classDistribution = defaultdict(list)
    for i in range(0, len(data)):
        className = data[i][14]

        if className not in classDistribution:

            classDistribution[className] = []
        classDistribution[className].append(data[i][:-1])

    return classDistribution


def summarizeDataSet(dataSet, option):
    """
    The function creates two dictionary with keys as attributes and
    values as the count of number of all possible values for each attribute and their
    corresponding counts.

    :param dataSet: the dictionary created in datCleanup.
    :return: greaterClassProbabilities : Dictionary for >50 class
    :return: smallerClassProbabilities : Dictionary for <=50 class
    """

    greaterClassProbablities = {}
    smallerClassProbablities ={}

    for key,value in dataSet.iteritems():
        age, workclass, finalweight, education, educationNum, maritalStatus, occupation, relationship, race, sex, \
        capitalGain, capitalLoss, hoursPW, nativeCountry = ([] for j in range(14))

        for i in range(0, len(value)):
            age.append(int(value[i][0]))
            workclass.append(value[i][1])
            finalweight.append(int(value[i][2]))
            education.append(value[i][3])
            educationNum.append(int(value[i][4]))
            maritalStatus.append(value[i][5])
            occupation.append(value[i][6])
            relationship.append(value[i][7])
            race.append(value[i][8])
            sex.append(value[i][9])
            capitalGain.append(int(value[i][10]))
            capitalLoss.append(int(value[i][11]))
            hoursPW.append(int(value[i][12]))
            nativeCountry.append(value[i][13])

        if key == ">50K":
            greaterClassProbablities['classProb'] = int(len(value))
            greaterClassProbablities['workclass'] = Counter(workclass)
            greaterClassProbablities['education'] = Counter(education)
            greaterClassProbablities['maritalStatus'] = Counter(maritalStatus)
            greaterClassProbablities['occupation'] = Counter(occupation)
            greaterClassProbablities['relationship'] = Counter(relationship)
            greaterClassProbablities['race'] = Counter(race)
            greaterClassProbablities['sex'] = Counter(sex)
            greaterClassProbablities['nativeCountry'] = Counter(nativeCountry)
            if option == 1:
                greaterClassProbablities['age'] = Counter(age)
                greaterClassProbablities['fnlwgt'] = Counter(finalweight)
                greaterClassProbablities['educationNum'] = Counter(educationNum)
                greaterClassProbablities['capitalGain'] = Counter(capitalGain)
                greaterClassProbablities['capitalLoss'] = Counter(capitalLoss)
                greaterClassProbablities['hoursPW'] = Counter(hoursPW)
            elif option == 2:
                greatDict = probablityFunction(value)
                greaterClassProbablities['prob'] =greatDict

        elif key == "<=50K":

            smallerClassProbablities['classProb'] = int(len(value))
            smallerClassProbablities['workclass'] = Counter(workclass)
            smallerClassProbablities['education'] = Counter(education)
            smallerClassProbablities['maritalStatus'] = Counter(maritalStatus)
            smallerClassProbablities['occupation'] = Counter(occupation)
            smallerClassProbablities['relationship'] = Counter(relationship)
            smallerClassProbablities['race'] = Counter(race)
            smallerClassProbablities['sex'] = Counter(sex)
            smallerClassProbablities['nativeCountry'] = Counter(nativeCountry)
            if option == 1:
                smallerClassProbablities['age'] = Counter(age)
                smallerClassProbablities['fnlwgt'] = Counter(finalweight)
                smallerClassProbablities['educationNum'] = Counter(educationNum)
                smallerClassProbablities['capitalGain'] = Counter(capitalGain)
                smallerClassProbablities['capitalLoss'] = Counter(capitalLoss)
                smallerClassProbablities['hoursPW'] = Counter(hoursPW)
            elif option == 2:
                smallDict = probablityFunction(value)
                smallerClassProbablities['prob'] = smallDict
    else: pass

    return greaterClassProbablities, smallerClassProbablities


def laplacianCorrection(lClass, recordValue, field, classLength):
    """
    The function is called from calculateProb function when there is a value for an attribute
    for which the probability is 0. The function makes a new entry in the corresponding attribute
    and change its count to 1. Then it adds 1 to the count of all the other already existing
    attributes and increases the class size by the total number of increments made.

    :param lClass: The class which has a new value for attribute
    :param recordValue: The new value
    :param field: The attribute/ key
    :param classLength: Length of class
    :return: lClass : The updated dictionary
    :return: classLength : updated length
    """

    i = 0
    if recordValue not in lClass[field]:
        for k in lClass[field]:
            lClass[field][k] += 1
            i += 1
        lClass[field][recordValue] = 1
    classLength += i+1
    return lClass, classLength


def probablityFunction(records):
    """
    Implementation of normal distribution function probability calculation.
    :param records:
    :return: dict
    """
    age, finalweight, educationNum, capitalGain, capitalLoss, hoursPW = ([] for j in range(6))

    for i in range(len(records)):
        record = data[i]
        age.append(int(record[0]))
        finalweight.append(int(record[2]))
        educationNum.append(int(record[4]))
        capitalGain.append(int(record[10]))
        capitalLoss.append(int(record[11]))
        hoursPW.append(int(record[12]))

    meanAge = float(mean(age))
    stdAge = float(stdev(age))

    meanFnlwgt = float(mean(finalweight))
    stdFnlwgt = float(stdev(finalweight))
    varFnlwgt = float(variance(finalweight))
    meanEN = float(mean(educationNum))
    stdEN = float(stdev(educationNum))
    varEN = float(variance(educationNum))
    meanCG = float(mean(capitalGain))
    stdCG = float(stdev(capitalGain))
    varCG = float(variance(capitalGain))
    meanCL = float(mean(capitalLoss))
    stdCL = float(stdev(capitalLoss))
    varCL = float(variance(capitalLoss))
    meanHPW = float(mean(hoursPW))
    stdHPW = float(stdev(hoursPW))
    varHPW = float(variance(hoursPW))
    dict = {'age': {'mean': meanAge, 'std': stdAge},
            'educationNum': {'mean': meanEN, 'std': stdEN, 'var': varEN},
            'fnlwgt': {'mean': meanFnlwgt, 'std': stdFnlwgt, 'var': varFnlwgt},
            'capitalGain': {'mean': meanCG, 'std': stdCG, 'var': varCG},
            'capitalLoss': {'mean': meanCL, 'std': stdCL, 'var': varCL},
            'hoursPW': {'mean': meanHPW, 'std': stdHPW, 'var': varHPW}}

    return dict


def normalDist(value, classProb):
    """
    Normal Distribution probability calculation.

    :param value:
    :param classProb:
    :return:
    """

    mean = classProb['mean']
    stdD = classProb['std']
    var = float(stdD) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(value) - float(mean)) ** 2 / (2 * var))
    prob = num / denom
    print prob
    return prob


def calculateProb(record, gClass, sClass, option):
    """
    The function calculates the probabilities for each record item.

    :param record:
    :param gClass:
    :param sClass:
    :return:
    """

    for i in range(len(record)-1):

        classGreater = gClass['classProb']
        classSmaller = sClass['classProb']
        pClassGreater = classGreater/(classGreater+classSmaller)
        pClassSmaller = classSmaller/(classSmaller+classGreater)

        if i ==0:
            field = "age"
            if option == 1:
                if record[i] not in gClass['age']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['age']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
                gAge = float(gClass['age'][record[i]]/ classGreater)
                sAge = float((sClass['age'][record[i]])/ classSmaller)
            else:
                probValG = gClass['prob']['age']
                probValS = sClass['prob']['age']
                gAge = normalDist(record[i], probValG)
                sAge = normalDist(record[i], probValS)


        elif i == 1:
            field = "workclass"
            if record[i] not in gClass['workclass']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['workclass']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gWrkclass = float(gClass['workclass'][record[i]]/classGreater)
            sWrkclass = float(sClass['workclass'][record[i]]/classSmaller)


        elif i == 2:
            field = "fnlwgt"
            if option == 1:
                if record[i] not in gClass['fnlwgt']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['fnlwgt']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
                gfnlwgt = float(gClass['fnlwgt'][record[i]]/classGreater)
                sfnlwgt = float(sClass['fnlwgt'][record[i]]/classSmaller)
            else:
                probValG = gClass['prob']['fnlwgt']
                probValS = sClass['prob']['fnlwgt']
                gfnlwgt = normalDist(record[i], probValG)
                sfnlwgt = normalDist(record[i], probValS)


        elif i == 3:
            field = "education"
            if record[i] not in gClass['education']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['education']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gEducation = float(gClass['education'][record[i]]/classGreater)
            sEducation = float(sClass['education'][record[i]]/classSmaller)

        elif i == 4:
            field = "educationNum"
            if option == 1:
                if record[i] not in gClass['educationNum']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['educationNum']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
                gEdnum = float(gClass['educationNum'][record[i]]/classGreater)
                sEdnum = float(sClass['educationNum'][record[i]]/classSmaller)
            else:

                probValG = gClass['prob']['educationNum']
                probValS = sClass['prob']['educationNum']
                gEdnum = normalDist(record[i], probValG)
                sEdnum = normalDist(record[i], probValS)

        elif i == 5:
            field = "maritalStatus"
            if record[i] not in gClass['maritalStatus']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['maritalStatus']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gMarital = float(gClass['maritalStatus'][record[i]]/classGreater)
            sMarital = float(sClass['maritalStatus'][record[i]]/classSmaller)

        elif i == 6:
            field = "occupation"
            if record[i] not in gClass['occupation']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['occupation']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gOccupation = float(gClass['occupation'][record[i]]/classGreater)
            sOccupation = float(sClass['occupation'][record[i]]/classSmaller)

        elif i == 7:
            field = "relationship"
            if record[i] not in gClass['relationship']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['relationship']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gRel = float(gClass['relationship'][record[i]]/classGreater)
            sRel = float(sClass['relationship'][record[i]]/classSmaller)

        elif i == 8:
            field = "race"
            if record[i] not in gClass['race']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['race']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gRace = float(gClass['race'][record[i]]/classGreater)
            sRace = float(sClass['race'][record[i]]/classSmaller)

        elif i == 9:
            field = "sex"
            if record[i] not in gClass['sex']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['sex']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gSex = float(gClass['sex'][record[i]]/classGreater)
            sSex = float(sClass['sex'][record[i]]/classSmaller)

        elif i == 10:
            field = "capitalGain"
            if option == 1:
                if record[i] not in gClass['capitalGain']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['capitalGain']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)

                gCG = float(gClass['capitalGain'][record[i]]/classGreater)
                sCG = float(sClass['capitalGain'][record[i]]/classSmaller)
            else:
                probValG = gClass['prob']['capitalGain']
                probValS = sClass['prob']['capitalGain']
                gCG = normalDist(record[i], probValG)
                sCG = normalDist(record[i], probValS)

        elif i == 11:
            field = "capitalLoss"
            if option == 1:

                if record[i] not in gClass['capitalLoss']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['capitalLoss']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
                gCL = float(gClass['capitalLoss'][record[i]]/classGreater)
                sCL = float(sClass['capitalLoss'][record[i]]/classSmaller)
            else:

                probValG = gClass['prob']['capitalLoss']
                probValS = sClass['prob']['capitalLoss']
                gCL = normalDist(record[i], probValG)
                sCL = normalDist(record[i], probValS)

        elif i ==12:
            field = "hoursPW"
            if option == 1:
                if record[i] not in gClass['hoursPW']:
                    gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
                elif record[i] not in sClass['hoursPW']:
                    sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
                gHPW = float(gClass['hoursPW'][record[i]]/classGreater)
                sHPW = float(sClass['hoursPW'][record[i]]/classSmaller)
            else:

                probValG = gClass['prob']['hoursPW']
                probValS = sClass['prob']['hoursPW']
                gHPW = normalDist(record[i], probValG)
                sHPW = normalDist(record[i], probValS)

        elif i == 13:
            field = "nativeCountry"
            if record[i] not in gClass['nativeCountry']:
                gClass, classGreater = laplacianCorrection(gClass, record[i], field, classGreater)
            elif record[i] not in sClass['nativeCountry']:
                sClass, classSmaller = laplacianCorrection(sClass, record[i], field, classSmaller)
            gNativeC = float(gClass['nativeCountry'][record[i]]/classGreater)
            sNativeC = float(sClass['nativeCountry'][record[i]]/classSmaller)

    gprobablity = float(gAge * gWrkclass * gfnlwgt * gEducation * gEdnum * gMarital * gOccupation * gRel * gRace * gSex
                        * gCG * gCL * gHPW * gNativeC * pClassGreater)
    sProbablity = float(sAge * sWrkclass * sfnlwgt * sEducation * sEdnum * sMarital * sOccupation * sRel * sRace * sSex
                        * sCG * sCL * sHPW * sNativeC * pClassSmaller)

    if gprobablity > sProbablity:
        predict = ">50K"
    elif sProbablity > gprobablity:
        predict = "<=50K"

    return predict


def predict(testData, greaterClass, smallerClass, option):
    """
    The function makes prediction of the testing dataset on the basis of probablities calculated
    on training dataset. For probability computation, function calls calculateProb function which
    calculate probabilities and return the predicted class. The function then checks the actual class
    and enter actual and predict in a evaluation matrix.

    :param testData: Testing dataset
    :param greaterClass: attribute values of ">50K"
    :param smallerClass: attribute values of "<=50K"
    :return: evluation matrix
    """

    evaluation = []
    for i in range(len(testData)):
        classPredict = calculateProb(testData[i], greaterClass, smallerClass, option)
        actualClass = testData[i][14]
        evaluation.append([actualClass, classPredict])
    return evaluation


def writeResult(dataSet):
    """
    Writes evaluation matrix.
    
    :param dataSet:
    :return:
    """

    with open('predictionResult.json', 'w') as outfile:
        json.dump(dataSet, outfile)


if __name__ == "__main__":

    start = time.time()

    accuracyData = []
    data = readData()
    option = askOption()
    if option == 1:
        dataset = equalWidthBinning(data)
        kfoldData = kFoldCrossValidation(dataset)
    else:
        kfoldData = kFoldCrossValidation(data)
    for i in range(10):
        trainFold = []
        testFold = kfoldData.pop(i)
        for j in range(0, len(kfoldData)):
            trainFold += kfoldData[j]
        print("trainfold: "+ str(len(trainFold)))
        dataToClassify = dataCleanup(trainFold)
        greaterClass, smallerClass = summarizeDataSet(dataToClassify, option)
        prediction = predict(testFold, greaterClass, smallerClass, option)
        accuracyData.append(prediction)
        kfoldData.append(testFold)

    writeResult(accuracyData)
    print("Class predictions have been written in the file predictionResult.json.")
    print("Execution time: %s" %(time.time()-start))