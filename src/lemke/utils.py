# file utilities

import fractions

import numpy as np

# global constants, mutable
# https://stackoverflow.com/questions/1977362/how-to-create-module-wide-variables-in-python
DEFAULT_DECIMALS = 4
MAXDECIMALS = 20
# roundingwarn = False


commentchars = "#%*"  # lines starting with these are ignored


def validate_decimals(decimals):
    if not isinstance(decimals, int):
        raise TypeError("decimals must be an integer")
    if decimals < 0 or decimals > MAXDECIMALS:
        raise ValueError(
            f"{decimals} as number of decimals not in allowed range 0 to {MAXDECIMALS}"
        )


# read file into list of line-strings
# truncate leading and trailing blanks
# ignore blank lines and lines starting with commentchars
def stripcomments(filename):
    # http://stackoverflow.com/questions/12330522/reading-a-file-without-newlines
    newlist = []
    with open(filename) as temp:
        temp = temp.read().splitlines()
        # strip comments
        for line in temp:
            line = line.strip()
            if line != "" and line[0] not in commentchars:
                newlist.append(line)
    return newlist


# convert lines to words
def towords(lines):
    words = []
    for line in lines:
        ell = line.split()
        for w in ell:
            words.append(w)
    return words


# convert s to fraction
# if s contains ".": convert to decimal fraction
# (numerator deciDenom)
def tofraction(s, decimals):
    deciDenom = 10 ** decimals
    if isinstance(s, str) and "." in s:
        s = float(s)
    if isinstance(s, float):
        num = int(abs(s) * deciDenom + 0.5)  # round .5 away from zero
        if s < 0:
            num = -num
        return fractions.Fraction(num, deciDenom)
    # any other s than a float or string containing '.':
    return fractions.Fraction(s)


# create n-vector of fractions from words[start,start+n)
def tovector(n, words, start, decimals):
    vector = np.zeros(n, dtype=fractions.Fraction)
    for i in range(n):
        vector[i] = tofraction(words[start + i], decimals)
    return vector


# create (m,n)-matrix of fractions from words[start,start+m*n)
def tomatrix(m, n, words, start, decimals):
    C = np.zeros((m, n), dtype=fractions.Fraction)
    k = start
    for i in range(m):
        for j in range(n):
            C[i][j] = tofraction(words[k], decimals)
            k += 1
    return C
