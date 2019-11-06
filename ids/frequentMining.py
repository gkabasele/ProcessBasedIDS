import pyfpgrowth as fp

"""
Example geeksforgeeks
E=1, K=2, M=3, N=4, O=5, Y=6
D=7, A=8, C=9, U=10, I=11
"""

transactions = [[1, 2, 3, 4, 5, 6],
                [7, 1, 2, 4, 5, 6],
                [8, 1, 2, 3],
                [9, 2, 3, 10, 6],
                [9, 1, 11, 2, 5]]

minimum_support = 3
patterns = fp.find_frequent_patterns(transactions, minimum_support)

print(patterns)
print("--------")

rules = fp.generate_association_rules(patterns, 0.9)

print(rules)
