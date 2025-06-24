input = input("Enter rock, paper, or scissors: ").lower()

import random
choices = ["rock", "paper", "scissors"]
computer_choice = random.choice(choices)
print(f"Computer chose: {computer_choice}")
if input == computer_choice:
    print("It's a tie!")
elif (input == "rock" and computer_choice == "scissors") or \
     (input == "paper" and computer_choice == "rock") or \
     (input == "scissors" and computer_choice == "paper"):
    print("You win!")
elif (input == "rock" and computer_choice == "paper") or \
     (input == "paper" and computer_choice == "scissors") or \
     (input == "scissors" and computer_choice == "rock"):
    print("You lose!")
else:
    print("Invalid input! Please enter rock, paper, or scissors.")
# This code implements a simple Rock, Paper, Scissors game where the user plays against the computer.
# The user inputs their choice, and the computer randomly selects one of the three options.
# The program then compares the choices and determines the outcome: win, lose, or tie.