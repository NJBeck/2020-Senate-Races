Using [Cook's Political Report's ratings for the 2020 Senate races](https://cookpolitical.com/ratings/senate-race-ratings/) and [their historical accuracy](https://cookpolitical.com/accuracy) to calculate the probabilities of different outcomes (e.g. Dems winning 15 of the 35 available seats). This is done in a rather brute force manner where every combination of outcome is considered and summed. For example, in the case of Dems winning 15 seats even the outcome of them winning the 12 Republican 'safe' seats and 3 Republican 'likely' seats and then losing all others is included. For every combination this is done as a product of binomical distribution pmf's binom(k, n, p) where n is the number of seats of that category, k is the number of seats won, and p is the probability of winning 1 seat. 

* There is an optional sanity check with a quick and simple monte carlo method which has conformed closely with each of my trials.
* Option to output this to a simple bar chart
* Can be easily reapplied to any other sort similarly categorized election map or field of outcomes

dependencies: numpy, matplotlib, scipy