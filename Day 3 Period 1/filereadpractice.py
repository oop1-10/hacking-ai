choice = input("Do you want to read a file? (y/n): ").lower()
global content
content = ""
global filename

if choice == 'y':
    filename = input("Enter a filename please: ")
    try:
        with open(filename, 'r') as file:
            content = file.read()
            print("File content:")
            print(content)
    except FileNotFoundError:
        print("File not found! Please check the filename and try again.")
else:
    filename = input("Enter the filename you want to create: ")
    print("Enter content below (enter 'done' when finished): \n")
    while content != "done":
        content = input("")
        if content.lower() != "done":
            try:
                with open(filename, 'a') as file:
                    file.write(content + "\n")
            except Exception as e:
                print("Error creating file:", e)

choice = input("Do you want to edit this file? (y/n): ").lower()

if choice == 'y':
    while True:
        print("Current content of the file: ")
        with open(filename, 'r') as file:
            print(file.read())
        row = input("Enter the row you want to edit (or 'done' to finish): ")
        if row.lower() == 'done':
            print("Exiting program...")
            break
        else:
            try:
                with open(filename, 'r') as file:
                    lines = file.readlines()

                    print("Current row content: ", lines[int(row)-1])

                edit_content = input("What would you like to change it to? ")
                lines[int(row)-1] = edit_content + "\n"

                with open(filename, 'w') as file:
                    file.writelines(lines)
            except Exception as e:
                print("Unspecified File Error:", e)
else:
    print("Exiting program without editing.")
