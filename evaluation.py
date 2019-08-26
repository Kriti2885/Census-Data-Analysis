"""

@Author : Kriti Upadhyaya

@Order of Execution -3

@Description:
The program evaluates naive Bayes algorithm prediction by calculating accuracy,
precision, recall, F1 and MCC . The program also generates the confusion matrix.
"""

from __future__ import division
import json
import math


def readData():
    '''
    The function reads data file which is created by dataPreProcess program.
    The data file now does'nt have any missing value. Missing values have been taken care of by your choice of option.
    :return: dataSet
    '''
    result =[]
    with open('predictionResult.json') as f:
        data = json.load(f)
    for i in range(len(data)):
        result += data[i]
    return result


def evaluateClassification(evaluationData):
    """
    Evaluates accuracy.
    :param evaluationData:
    :return: accuarcy
    """

    accuracyPrediction = 0
    for i in range(len(evaluationData)):
        if evaluationData[i][0] == evaluationData[i][1]:
            accuracyPrediction += 1

    accuracy = float(accuracyPrediction/ len(evaluationData))
    return accuracy


def confusionMatrix(evaluationData):
    """
    Creates Confusion matrix by calculating true Positive, true Negative, false Positive, false Negative.
    :param evaluationData:
    :return: truePos, trueNeg, falsePos, falseNeg
    """

    truePos, trueNeg, falsePos, falseNeg = (0 for k in range(4))

    for i in range(len(evaluationData)):
        if evaluationData[i][0] == "<=50K" and evaluationData[i][1] == "<=50K":
            truePos += 1
        elif evaluationData[i][0] == "<=50K" and evaluationData[i][1] == ">50K":
            falseNeg += 1
        elif evaluationData[i][0] == ">50K" and evaluationData[i][1] == "<=50K":
            falsePos += 1
        elif evaluationData[i][0] == ">50K" and evaluationData[i][1] == ">50K":
            trueNeg += 1
    print("TruePos: %s, TrueNeg: %s, FalsePos: %s, FalseNeg: %s" % (truePos, trueNeg, falsePos, falseNeg))
    return truePos, trueNeg, falsePos, falseNeg


def calculateRecall(truePos, falseNeg):
    """
    Calculates Recall.
    :param truePos:
    :param falseNeg:
    :return: recall
    """
    recall = truePos/(truePos+falseNeg)
    return recall


def calculatePrecision(truePos, falsePos):
    """
    Calculates Precision.
    :param truePos:
    :param falsePos:
    :return: precision
    """

    precision = truePos/(truePos + falsePos)
    return precision


def calculateF1(precision, recall):
    """
    Calculates F1 Score.
    :param precision:
    :param recall:
    :return: f1Measure
    """

    f1Measure = 2*(precision*recall)/ (precision + recall)
    return f1Measure


def calculateMCC(truePos, trueNeg, falsePos, falseNeg):
    """
    Calculation of MCC.
    :param truePos:
    :param trueNeg:
    :param falsePos:
    :param falseNeg:
    :return: mccMeasure
    """
    num = ((truePos * trueNeg) - (falsePos * falseNeg))
    denom = ((truePos + falsePos) * (truePos + falseNeg)* (trueNeg + falsePos) * (trueNeg + falseNeg))
    newDenom = math.sqrt(denom)
    mccMeasure = num/ newDenom
    return mccMeasure


if __name__ == "__main__":
    evaluationData = readData()
    accuracy = evaluateClassification(evaluationData)
    print("Accuracy: ") + str(accuracy)
    truePos, trueNeg, falsePos, falseNeg = confusionMatrix(evaluationData)
    precision = calculatePrecision(truePos, falsePos)
    recall = calculateRecall(truePos, falseNeg)
    f1Measure = calculateF1(precision, recall)
    mccMeasure = calculateMCC(truePos, trueNeg, falsePos, falseNeg)
    print("Precision: " + str(precision) + ", Recall: " + str(recall) + " and f1 measure is: " + str(f1Measure))
    print mccMeasure

