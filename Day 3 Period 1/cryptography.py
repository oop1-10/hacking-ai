# Ceasar Cipher in Python

choice = int(input("Do you want to encrypt or decrypt? 1 = encrypt, 2 = decrypt: "))

# Option to read from file
input_choice = input("Read from file? (y/n): ").lower()

if input_choice == 'y':
    filename = input("Enter filename: ")
    try:
        with open(filename, 'r') as file:
            word = file.read().strip()
        print(f"Read from file: {word}")
    except FileNotFoundError:
        print("File not found! Using manual input instead.")
        word = input("Please enter a word: ")
else:
    word = input("Please enter a word: ")

shift = int(input("Please enter the shift value: "))

# OLD CODE
if choice == 1:
    ans = ""
    for i in range (len(word)):
        char = word[i]

        if char.isupper():
            ans += chr((ord(char) + shift - 65) % 26 + 65)
        else:
            ans += chr((ord(char) + shift - 97) % 26 + 97)

    result = "The encrypted word is: " + ans
    print(result)
else:
    ans = ""
    for i in range (len(word)):
        char = word[i]

        if char.isupper():
            ans += chr((ord(char) - shift - 65) % 26 + 65)
        else:
            ans += chr((ord(char) - shift - 97) % 26 + 97)
    
    result = "The decrypted word is: " + ans
    print(result)
# OLD CODE

# Option to save result to file
save_choice = input("Save result to file? (y/n): ").lower()
if save_choice == 'y':
    output_file = input("Enter output filename: ")
    try:
        with open(output_file, 'w') as file:
            file.write(f"Original: {word}\n")
            file.write(f"Shift: {shift}\n")
            file.write(f"{result}\n")
        print(f"Result saved to {output_file}")
    except Exception as e:
        print(f"Error saving to file: {e}")

