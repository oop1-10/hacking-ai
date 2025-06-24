list = [1, 2, 3, 4, 5]

input = input("Enter the number you want to find: ")

for i in range(len(list)):
    if list[i] == int(input):
        print(f"The number {input} is found at index {i}.")
        break