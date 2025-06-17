height = int(input("Enter the height of the pyramid: "))

for i in range(1, height + 1):
    print(" " * (height - i) + "*" * (2 * i - 1))
    