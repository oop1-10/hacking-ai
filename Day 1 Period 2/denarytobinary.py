num = input("Enter a denary(decimal) number: ")
binary_num = bin(int(num))[2:]
print(f"The binary equivalent of {num} is {binary_num}.")

def decimal_to_binary(n):
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append(str(n % 2))
        n //= 2
    return "".join(reversed(digits))

# Manual conversion without bin()
num_int = int(num)
manual_binary = decimal_to_binary(num_int)
print(f"The binary equivalent (manual) of {num} is {manual_binary}.")