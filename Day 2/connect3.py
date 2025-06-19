ROWS = 6
COLS = 7
CONNECT_N = 3

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * (COLS * 4 - 1))
    print("  ".join(str(i+1) for i in range(COLS)))

def check_win(board, player):
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLS - CONNECT_N + 1):
            if all(board[row][col + i] == player for i in range(CONNECT_N)):
                return True
    # Check vertical
    for col in range(COLS):
        for row in range(ROWS - CONNECT_N + 1):
            if all(board[row + i][col] == player for i in range(CONNECT_N)):
                return True
    # Check diagonal (down-right)
    for row in range(ROWS - CONNECT_N + 1):
        for col in range(COLS - CONNECT_N + 1):
            if all(board[row + i][col + i] == player for i in range(CONNECT_N)):
                return True
    # Check diagonal (up-right)
    for row in range(CONNECT_N - 1, ROWS):
        for col in range(COLS - CONNECT_N + 1):
            if all(board[row - i][col + i] == player for i in range(CONNECT_N)):
                return True
    return False

def check_draw(board):
    return all(cell != ' ' for row in board for cell in row)

def get_next_open_row(board, col):
    for row in range(ROWS - 1, -1, -1):  # Start from bottom row
        if board[row][col] == ' ':
            return row
    return None

def main():
    board = [[' ' for _ in range(COLS)] for _ in range(ROWS)]
    current_player = 'X'
    while True:
        print_board(board)
        print(f"Player {current_player}'s turn.")
        try:
            col = int(input(f"Enter column (1-{COLS}): ")) - 1
            if col not in range(COLS):
                print(f"Invalid input. Please enter a column from 1 to {COLS}.")
                continue
            row = get_next_open_row(board, col)
            if row is None:
                print("That column is full. Try another one.")
                continue
            board[row][col] = current_player
            if check_win(board, current_player):
                print_board(board)
                print(f"Player {current_player} wins!")
                break
            if check_draw(board):
                print_board(board)
                print("It's a draw!")
                break
            current_player = 'O' if current_player == 'X' else 'X'
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
