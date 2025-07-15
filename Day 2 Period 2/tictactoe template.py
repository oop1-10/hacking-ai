board = [[' ' for i in range(3)] for j in range(3)]
turnCount = 0

def print_board(board):
    for i in range(3):
        for j in range(3):
            print(board[i][j], end=' | ')
        print("\n")

def check_winner(board): # THIS FUNCTION SHOULD RETURN 'X' (X WON), 'O' (O WON), OR FALSE (NO WIN YET)
    # Check rows
    for i in range(3):
        #ENTER IF HERE

    for i in range(3):
        # Check columns
        #ENTER IF HERE  
    
    # IGNORE ANY ERRORS HERE
    # Check diagonals
    #ENTER 2 DIFFERENT IF'S HERE

    return False

while True:
    print_board(board)
    print("Player", "X" if turnCount % 2 == 0 else "O", "'s turn")
    row = int(input("Enter row (1-3): ")); col = int(input("Enter column (1-3): "))

    turnCount += 1


    # CHECK THE TURN COUNT AND PLACE "MARKER" HERE ('X' or 'O')
    # IF YOU ENTER AN ALREADY TAKEN CELL, -1 FROM THE TURN COUNT AND TRY AGAIN
    # IF, ELIF, ELSE REQUIRED HERE


    # FINALLY CHECK FOR A WINNER (IF CHECK WINNER RETURNS 'X' OR 'O'), ADD AN ELIF THAT CHECKS FOR A DRAW (turnCount >= 9)
