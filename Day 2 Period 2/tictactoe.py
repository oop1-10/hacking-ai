board = [[' ' for i in range(3)] for j in range(3)]
turnCount = 0

def print_board(board):
    for i in range(3):
        print("| ", end="")
        for j in range(3):
            print(board[i][j], end=' | ')
        print("\n")

def check_winner(board):
    # Check rows
    for i in range(3):
        if (all(board[i][j] == 'X' for j in range(3)) or all(board[i][j] == 'O' for j in range(3))):
            return board[i][0]

    for i in range(3):
        # Check columns
        if (all(board[j][i] == 'X' for j in range(3)) or all(board[j][i] == 'O' for j in range(3))):
            return board[0][i]
    

    if (all(board[i][i] == 'O' for i in range(3)) or all(board[i][i] == 'X' for i in range(3))):
        return board[1][1]
    # Check diagonals
    if (all(board[i][2-i] == 'O' for i in range(3)) or all(board[i][2-i] == 'X' for i in range(3))):
        return board[1][1]

    return False

while True:
    print_board(board)
    print("Player", "X" if turnCount % 2 == 0 else "O", "'s turn")
    row = int(input("Enter row (1-3): ")); col = int(input("Enter column (1-3): "))

    if (row < 1 and row > 3) or (col < 1 and col > 3):
        print("Invalid input, please enter numbers between 1 and 3.")
        continue
    else:
        turnCount += 1

        selectedRow = row - 1
        selectedCol = col - 1

        if board[selectedRow][selectedCol] == ' ' and turnCount % 2 == 1:
            board[selectedRow][selectedCol] = 'X'  # Player's move
        elif board[selectedRow][selectedCol] == ' ' and turnCount % 2 == 0:
            board[selectedRow][selectedCol] = 'O'
        else:
            print("Cell already taken, try again.")
            turnCount-=1
            continue

        if check_winner(board) == 'X' or check_winner(board) == 'O':
            print_board(board)
            winner = check_winner(board)
            print("Player", winner, "wins!")
            break
