user_input = input("How can I assist you today? ").lower()

if user_input == "hello":
    print("Hello! How can I help you?")
elif user_input == "hi":
    print("Hello! How can I help you?")
elif user_input == "help":
    print("I'm here to help! What do you need?")
elif user_input == "how tall is the cn tower":
    print("The CN Tower is 553.3 meters tall.")
elif user_input == "bye":
    print("Goodbye!")
elif user_input.isdecimal():
    print(f"The binary equivalent of {user_input} is {bin(int(user_input))[2:]}.")
else:
    print("I don't understand that.")
