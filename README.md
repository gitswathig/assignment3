# assignment3
#game1
import math
import random

class TicTacToe:

    def __init__(self):
        self.board = [' ' for _ in range(9)]  
        self.current_winner = None 

    def print_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = math.floor(square / 3)
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([s == letter for s in diagonal2]):
                return True
        return False

    def minimax(self, depth, maximizingPlayer):
        if self.current_winner:
            return 1 if self.current_winner == 'X' else -1
        
        if maximizingPlayer:
            bestScore = -math.inf
            for move in self.available_moves():
                self.make_move(move, 'X')
                score = self.minimax(depth+1, False)
                self.board[move] = ' '
                bestScore = max(score, bestScore)
            return bestScore
        else:
            bestScore = math.inf
            for move in self.available_moves():
                self.make_move(move, 'O')
                score = self.minimax(depth+1, True)
                self.board[move] = ' '
                bestScore = min(score, bestScore)
            return bestScore

    def get_best_move(self, letter):
        bestScore = -math.inf
        bestMove = 0
        for move in self.available_moves():
            self.make_move(move, letter)
            score = self.minimax(0, letter=='O')
            self.board[move] = " "
            if score > bestScore:
                bestScore = score
                bestMove = move
        return bestMove

def play(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player(game)
        else:
            square = x_player(game)
        if game.make_move(square, letter):
 
            if print_game:
                print(letter + ' makes a move to square {}'.format(square))
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                return letter 

            letter = 'O' if letter == 'X' else 'X'

        if print_game:
            print('It\'s a tie!')

if __name__ == '__main__':
    x_player = lambda x: random.choice(x.available_moves())
    o_player = lambda x: x.get_best_move('O') 
    t = TicTacToe()
    play(t, x_player, o_player, print_game=True)

    
# game2
import random

print("Welcome to the number guessing game!")
print("I'm thinking of a number between 1 and 100.")

# Generate random number between 1 and 100
secret_num = random.randint(1, 100)

# Print different messages based on number of guesses left
attempts = 10
while attempts > 0:
    print(f"You have {attempts} attempts remaining to guess the number.")

    # Get user's guess 
    guess = int(input("Make a guess: "))

    # Check if guess is correct
    if guess == secret_num:
        print(f"Congratulations! You guessed the number {secret_num} correctly!")
        break
    
    # Give hint if guess is too low or high
    if guess < secret_num:
        print("Your guess is too low. Try again.")
    elif guess > secret_num:
        print("Your guess is too high. Try again.")

    attempts -= 1

# Game over message    
if attempts == 0:
    print(f"The number was {secret_num}. Better luck next time!")
