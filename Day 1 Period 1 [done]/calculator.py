try:
        a = float(input("Enter first number: "))
        op = input("Enter operator (+, -, *, /): ").strip()
        b = float(input("Enter second number: "))
except ValueError:
        print("Invalid number entered.")

if op == "+":
    result = a + b
elif op == "-":
    result = a - b
elif op == "*":
    result = a * b
elif op == "/":
    if b == 0:
        print("Error: Division by zero.")
    result = a / b
else:
    print(f"Unsupported operator '{op}'.")

print(f"Result: {result}")