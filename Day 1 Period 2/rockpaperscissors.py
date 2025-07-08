import random

userInput = ""

while userInput != "EXIT" or "exit":
    userInput = input("Enter rock, paper, or scissors (or type EXIT to quit): ")
    randomNumber = random.randint(1,3)

    if userInput == "EXIT" or userInput == "exit":
        print("Thanks for playing!")
        break
    elif userInput in ["rock", "paper", "scissors"]:
        if randomNumber == 1:
            computer_choice = "rock"
        elif randomNumber == 2:
            computer_choice = "paper"
        else:
            computer_choice = "scissors"

        print(f"Computer chose: {computer_choice}")

        if userInput == computer_choice:
            print("Tie!")
        elif userInput == "rock" and computer_choice == "scissors":
            print("You win!")
        elif userInput == "paper" and computer_choice == "rock":
            print("You win!")
        elif userInput == "scissors" and computer_choice == "paper":
            print("You win!")
        else:
            print("You lose!")
    else:
        print("Invalid input. Please enter rock, paper, or scissors.")