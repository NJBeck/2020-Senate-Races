import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import binom

solidD = 8
likelyD = 2
leanD = 1
TossupD = 1
TossupR = 3
leanR = 1
likelyR = 7
solidR = 12
races = [solidD, likelyD, leanD, TossupD, TossupR, leanR, likelyR, solidR]

seatTotal = sum(races)      # total number of races
# num of races needed for 50/50 seats
toWin = 15

probSolidD = .998
probLikelyD = .94
probLeanD = .877
probTossupD = .492
probTossupR = .45
probLeanR = .164
probLikelyR = .042
probSolidR = .003
probs = [probSolidD, probLikelyD, probLeanD, probTossupD,
         probTossupR, probLeanR, probLikelyR, probSolidR]


# making a sorted list of tuples by number of races with the same prob
# ex: [(0.003, 12), (0.998, 8), (0.042, 7), (0.45, 3),
#       (0.94, 2), (0.877, 1), (0.492, 1), (0.164, 1)]
# doing this so the seat totals and probs sort the same way
# then unzip this to get sorted race totals and their probibilities
raceTuples = [(probs[i], races[i]) for i in range(len(races))]
raceTuples.sort(key=lambda x: x[1], reverse=True)

raceTotals = np.array([race[1]for race in raceTuples])
raceProbs = np.array([prob[0] for prob in raceTuples])

'''
The idea is by dividing the races up into categories of same probibility
we can use the products of binomial distributions to get the probility
of a specific set of race outcomes
summing over all of these gives us the probability
of winning a certain number of races (such as the 15 needed for 50/50)
example: the prob of winning 7 solidD races = binom.pmf(7, 8, probSolidD)
[solidD=8, likelyD=2, leanD=1, TossupD=1,
TossupR=3, leanR=0, likelyR=0, solidR=0]
is one possible combination that adds up the the 15 needed for 50
that is the product of 8 binom.pmf()'s
summing over all possible combinations of 15 wins gives us the prob
of winning exactly 15 seats
'''

def _distribute(n, raceTotalArray, startArray):
    '''
    gives the permutations of arrays with n in every slot given that
    n is also less than the value in raceTotalArray in that slot
    ex: _distribute(2, [12,8,7,3,2,1,1,1], [0,0,0,3,0,0,0,0]) gives
    [array([2, 0, 0, 3, 0, 0, 0, 0]), 
    array([0, 2, 0, 3, 0, 0, 0, 0]), array([0, 0, 2, 3, 0, 0, 0, 0]),
    array([0, 0, 0, 3, 2, 0, 0, 0])]
    '''
    arrays = []
    for i in range(len(startArray)):
        if raceTotalArray[i] >= n:
            if startArray[i] == 0:               
                tempArray = np.copy(startArray)
                tempArray[i] = n
                arrays.append(tempArray)
        else:
            break
    return arrays

def find_sums(n, maxSize, maxLen):
    '''
    returns a list of lists whose elements add to the number given
    maxSize limits the largest element in a list
    maxLen is the limit of the length of any list
    example: find_sums(5, 3, 4) = [[3, 2], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1]]
    '''
    sums = []
    if n <= maxSize:
        sums.append([n])

    for i in range(1, n):
        # we iterate through splitting a number into a base and a residual
        # eg 5 can be split into 4 and 1 then 3 and 2
        base = n - i
        # discard those with bases too large to fit in our array
        if base <= maxSize:

            # repeat the process of splitting for the residual i
            # we now use a maxSize of base in order to limit repetition
            # e.g. 3 can be split to [2,1] but not [1,2]
            # because the residual 2 is larger than the base of 1
            residual = find_sums(i, base, maxLen - 1)
            # find_sums gives us a list of lists of combinations
            # we take each list and add the base to it
            if residual:
                for res in residual:
                    temp = [base] + res
                    # if the new list is shorter than maxLen we append it
                    if len(temp) <= maxLen:
                        sums.append(temp)

    return sums


def find_combs(n, raceTotalArray=raceTotals):
    '''
    The idea here is to take every combination of numbers that adds up to n
    subject to the constraints of a max num in the list and max list length.
    This is to account for a limited number of categories of races
    (i.e. leanR, leanD, etc.)
    Then we find every combination of those numbers which will fit into
    the array of races per category
    example: [12, 8] can only fit into our [12,8,7,3,2,1,1,1] in 1 way
    returns an array of every such combination
    '''
    finalArray = []
    # example of sumCombinations: [[3,2], [3,1,1], [2,2,1], [2,1,1,1]]
    sumsCombinations = find_sums(n, raceTotalArray[0], len(raceTotalArray))
    for comb in sumsCombinations:  # [3,2] then [3,1,1] etc
        # starts empty but holds the output of _distribute to iterate over
        # example: distributing the 3 over [0,0,0,0]
        # subject to the constraint [3,2,1,1] is just [3,0,0,0]
        # then 2 gets distributed over that which only gives [3,2,0,0]
        distributedArrays = [np.zeros(len(raceTotalArray), dtype=int)]
        for num in comb:   # example from above: 3 then 2 etc
            tempList = []
            for arr in distributedArrays:
                # take each array from a distribution then
                # then distribute the next num then store it in a temp list
                tempList += _distribute(num, raceTotalArray, arr)
            # after this has been done we replace the distributedArray
            # and distribute the next number
            distributedArrays = tempList
        # after this has been done for every numb in a combination
        # we add it to the finalArray
        finalArray += distributedArrays
    return np.unique(np.array(finalArray), axis=0)

def find_prob(n=15, raceTotalArray=raceTotals, raceProbArray=raceProbs, totalSeats=seatTotal):
    '''
    as described above the functions above we multiply the binom.pmf()
    of every number in each array in the array from find_combs
    we then do this for every number between n and total number
    of possible seat wins (i.e. all 35)
    '''
    finalProb = 0
    combs = find_combs(n, raceTotalArray)
    combProbArray = []
    for arr in combs:
        # take each array of combinations and turn them into probs from binom
        probList = []
        for i, raceCount in enumerate(arr):
            prob = binom.pmf(raceCount, raceTotalArray[i], raceProbArray[i])
            probList.append(prob)
        # multiply every binom.pmf in the resulting array append it
        combProbArray.append(np.prod(np.array(probList)))
    finalProb += np.sum(np.array(combProbArray))
    return finalProb

def sum_probs(n=15, raceTotalArray=raceTotals, raceProbArray=raceProbs, totalSeats=seatTotal):
    '''
    returns the prob of winning n or more races
    '''
    probSum = 0
    for total in range(n, totalSeats + 1):
        probSum += find_prob(total, raceTotalArray, raceProbArray, totalSeats)
    print(f'The probability of Dems winning {n} or more seats is {probSum}')
    return probSum

def prob_list(n=15, totalSeats=seatTotal, plot=False, printed=True):
    '''
    returns a list of tuples containing (seats won, probability)

    '''
    probList = []
    probSum = 0
    for i in range(totalSeats + 1):
        prob = find_prob(i)
        if i >= n:
            probSum += prob
        probList.append((i, prob))
        if printed == True:
            print(f'The probability of winning {i} seat(s) is {prob}')
    print(f'The probability of Dems winning {n} or more seats is {probSum}')
    if plot == True:
        x, y = zip(*probList)
        plt.bar(x, y)
        plt.xlabel('# of Dem seats won')
        plt.ylabel('probability')
        plt.show()
    return probList

def monte_carlo(testN=10000, plot=True, ToWin=15):
    # create a list of the probabilities of a category of races
    # each copied by the number of races of that of that type
    # example: there are 8 solidD races so 8 copies of .998
    seatProbList = []
    for _ in range(solidD):
        seatProbList.append(probSolidD)
    for _ in range(likelyD):
        seatProbList.append(probLikelyD)
    for _ in range(leanD):
        seatProbList.append(probLeanD)
    for _ in range(TossupD):
        seatProbList.append(probTossupD)
    for _ in range(TossupR):
        seatProbList.append(probTossupR)
    for _ in range(leanR):
        seatProbList.append(probLeanR)
    for _ in range(likelyR):
        seatProbList.append(probLikelyR)
    for _ in range(solidR):
        seatProbList.append(probSolidR)

    # we then copy that array by the test size
    ProbList = np.tile(seatProbList, (testN, 1))
    # we make an array of the same shape with random numbs between 0 and 1
    rands = np.random.random_sample((testN, len(seatProbList)))
    # whenever an entry in ProbList > the one in rands that is a victory
    successArray = np.greater(ProbList, rands)
    # adding up the number of successes in a single subarray
    # gives the number of successes in any trial which we store here
    successVector = np.sum(successArray, axis=1)
    # picking out the number of instances where successes > 15 is our answer
    successCount = np.sum(successVector >= ToWin)

    print(f"successfully won {ToWin}+ seats in " \
        + str(successCount/testN*100) + "% of " + str(testN) + ' trials')
    if plot == True:        
        plt.hist(successVector)
        plt.show()
    return successCount

if __name__ == '__main__':
    prob_list(plot=True)