numbers = [5, 2, 9, 1, 5, 6]

sorted_numbers = []
while numbers:
    min_value = numbers[0]
    for num in numbers:
        if num < min_value:
            min_value = num
    sorted_numbers.append(min_value)
    numbers.remove(min_value)
print("Sorted numbers:", sorted_numbers)
# This code sorts a list of numbers using a linear sort algorithm.
# It repeatedly finds the minimum value in the list, appends it to a new list,