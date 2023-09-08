#first program for machine learning

#ask numbers from user
print("Enter the numbers to be sorted: ")
a = [int(x) for x in input().split()]

#sort the numbers in the list
a.sort()

#print the sorted list
print(a)