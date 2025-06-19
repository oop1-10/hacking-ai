input = input("How can I assist you today? ").lower()

if input.isdecimal():
    binary_num = bin(int(input))[2:]
    print(f"The binary equivalent of {input} is {binary_num}.")
elif "hello" in input or "hi" in input:
    print("Hello! How can I help you?")
elif "help" in input:
    print("Sure, I'm here to help! What do you need assistance with?")
elif "how tall is the CN tower" in input:
    print("The CN Tower is approximately 553.3 meters tall.")
elif "bye" in input:
    print("Goodbye! Have a great day!")
else:
    print("I'm not sure how to respond to that.")
