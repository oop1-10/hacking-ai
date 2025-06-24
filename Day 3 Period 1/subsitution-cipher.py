# Encryption and Decryption for Substitution Cipher

all_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def encrypt(message):
    key = len(message) % len(all_letters)
    dict1 = {}
    for i in range(len(all_letters)):
        dict1[all_letters[i]] = all_letters[(i + key) % len(all_letters)]
    cipher_txt = ""
    for char in message:
        if char in all_letters:
            cipher_txt += dict1[char]
        else:
            cipher_txt += char
    return cipher_txt

def decrypt(cipher_txt):
    key = len(cipher_txt) % len(all_letters)
    dict2 = {}
    for i in range(len(all_letters)):
        dict2[all_letters[i]] = all_letters[(i - key) % len(all_letters)]
    decrypt_txt = ""
    for char in cipher_txt:
        if char in all_letters:
            decrypt_txt += dict2[char]
        else:
            decrypt_txt += char
    return decrypt_txt

# Interactive mode
choice = int(input("1 = encrypt, 2 = decrypt, 3 = demo: "))

if choice == 3:
    # Demo mode (original functionality)
    print("Demo - Encrypting: 'Hello how are you doing today'")
    encrypted = encrypt("Hello how are you doing today")
    print(f"Encrypted: {encrypted}")
    
    print("Demo - Decrypting: 'kHOOR KRZ DUH bRX GRLQJ WRGDb'")
    decrypted = decrypt('kHOOR KRZ DUH bRX GRLQJ WRGDb')
    print(f"Decrypted: {decrypted}")
    
else:
    # Option to read from file
    input_choice = input("Read from file? (y/n): ").lower()
    
    if input_choice == 'y':
        filename = input("Enter filename: ")
        try:
            with open(filename, 'r') as file:
                message = file.read().strip()
            print(f"Read from file: {message}")
        except FileNotFoundError:
            print("File not found! Using manual input instead.")
            message = input("Enter your message: ")
    else:
        message = input("Enter your message: ")
    
    if choice == 1:
        result = encrypt(message)
        operation = "encrypted"
    else:
        result = decrypt(message)
        operation = "decrypted"
    
    print(f"The {operation} message is: {result}")
    
    # Option to save result to file
    save_choice = input("Save result to file? (y/n): ").lower()
    if save_choice == 'y':
        output_file = input("Enter output filename: ")
        try:
            with open(output_file, 'w') as file:
                file.write(f"Original: {message}\n")
                file.write(f"Operation: {operation}\n")
                file.write(f"Result: {result}\n")
            print(f"Result saved to {output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")

