num = input("Enter a denary(decimal) number: ")
decimal_num = int(num)

# Manual binary conversion algorithm
if decimal_num == 0:
    binary_num = "0"
else:
    binary_num = ""
    temp = decimal_num
    
    # Divide by 2 repeatedly and collect remainders
    while temp > 0:
        remainder = temp % 2
        binary_num = str(remainder) + binary_num  # Add remainder to front
        temp = temp // 2  # Integer division by 2

print(f"The binary equivalent of {num} is {binary_num}.")

# Show the step-by-step process
print("\nStep-by-step conversion:")
temp = decimal_num
if temp == 0:
    print("0 ÷ 2 = 0 remainder 0")
else:
    steps = []
    while temp > 0:
        remainder = temp % 2
        quotient = temp // 2
        steps.append(f"{temp} ÷ 2 = {quotient} remainder {remainder}")
        temp = quotient
    
    for step in steps:
        print(step)
    
    print(f"Reading remainders from bottom to top: {binary_num}")