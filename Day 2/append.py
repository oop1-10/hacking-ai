numbers = [3, 7, 1, 9, 5]

print("Current list:", numbers)

new_number = int(input("Enter a number to add to the list: "))

numbers.append(new_number)
numbers.sort()

print("Updated list:", numbers)
