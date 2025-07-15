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
        if board[i][0] == board[i][1] == board[i][2] == 'X' or board[i][0] == board[i][1] == board[i][2] == 'O':
            return board[i][0]

    for i in range(3):
        # Check columns
        if board[0][i] == board[1][i] == board[2][i] == 'X' or board[0][i] == board[1][i] == board[2][i] == 'O':
            return board[0][i]
    

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == 'O' or board[0][0] == board[1][1] == board[2][2] == 'X':
        return board[1][1]
    if board[0][2] == board[1][1] == board[2][0] == 'O' or board[0][2] == board[1][1] == board[2][0] == 'X':
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

        if board[row-1][col-1] == ' ' and turnCount % 2 == 1:
            board[row-1][col-1] = 'X'  # Player's move
        elif board[row-1][col-1] == ' ' and turnCount % 2 == 0:
            board[row-1][col-1] = 'O'
        else:
            print("Cell already taken, try again.")
            continue

        if check_winner(board) == 'X' or check_winner(board) == 'O':
            print_board(board)
            winner = check_winner(board)
            print("Player", winner, "wins!")
            break