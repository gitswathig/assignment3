# assignment 1
import random

class Chatbot:
    def __init__(self):
        self.greetings = ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! What's on your mind?"]
        self.farewells = ["Goodbye! Have a great day!", "Farewell! Come back anytime.", "See you later!"]
        self.questions = [
            "What is your name?",
            "How can I assist you today?",
            "Do you have any specific questions?",
            "Tell me something interesting about yourself.",
            "Is there anything else I can help with?"
        ]
        self.informative_responses = [
            "That's interesting! Did you know that...",
            "I have some information related to that. Would you like to hear it?",
            "I can provide more details on that if you're interested."
        ]
        self.humorous_responses = [
            "Haha, good one! By the way...",
            "You've got a great sense of humor! Here's something amusing...",
            "Let me lighten the mood with a joke: Why did the chatbot go to therapy? To improve its AI-llignment!"
        ]

    def greet_user(self):
        print(random.choice(self.greetings))

    def ask_question(self, question):
        user_response = input(question + " ")
        self.respond_to_user_input(user_response)

    def respond_to_user_input(self, user_input):
        if any(word in user_input.lower() for word in ["bye", "exit", "quit"]):
            print(random.choice(self.farewells))
            exit()
        elif any(word in user_input.lower() for word in ["who", "what", "where", "when", "why", "how"]):
            print(random.choice(self.informative_responses))
        elif "?" in user_input:
            print("Interesting question! I'll have to think about that.")
        else:
            print(random.choice(self.humorous_responses))

    def handle_errors(self):
        print("Sorry, I didn't understand that. Can you please rephrase?")

    def start_conversation(self):
        self.greet_user()
        try:
            while True:
                random_question = random.choice(self.questions)
                self.ask_question(random_question)
        except KeyboardInterrupt:
            print("\nUser interrupted the conversation. Exiting.")
        except Exception as e:
            print(f"An error occurred: {e}")
            self.handle_errors()

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.start_conversation()
****
# assignment2
pip install xgboost
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score
data = pd.read_csv('dataset.csv')
data.head(10)
data.describe()
data.info()
categorical_cols = ['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere']
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])
    data.isnull().sum()
    data = data.drop('Patient ID', axis=1)
    plt.figure(figsize=(8, 7))
for col in categorical_cols:
    sns.countplot(data=data, x=col)
    plt.show()
    bp_split = data['Blood Pressure'].str.split('/', expand=True)
data['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(0).astype(int)
data['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(0).astype(int)
data.drop('Blood Pressure', axis=1, inplace=True)
plt.figure(figsize=(19, 10))
sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='Heart Attack Risk', y='Age', data=data)
plt.title('Age vs. Heart Attack Risk')
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(x='Heart Attack Risk', y='Cholesterol', data=data)
plt.title('Cholesterol vs. Heart Attack Risk')
plt.show()
X = data.drop('Heart Attack Risk', axis=1)
y = data['Heart Attack Risk'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision'])
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB()
}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=1)
    print(f'{name} - Accuracy: {accuracy}, Precision: {precision}')
    new_row = {'Model': name, 'Accuracy': accuracy, 'Precision': precision}
    results_df = pd.concat([results_df, pd.DataFrame([new_row])])
   rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test, rf_predictions)
roc_auc = roc_auc_score(y_test, rf_predictions)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
plt.figure(figsize=(12, 6))

# Accuracy ve Precision için iki ayrı çubuk grafik çizin
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Accuracy by Model')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Precision', data=results_df)
plt.title('Precision by Model')

plt.tight_layout()
plt.show()

print(results_df.columns)

for name, row in results_df.iterrows():
    print(f'{name} - Model: {row["Model"]}, Accuracy: {row["Accuracy"]}, Precision: {row["Precision"]}')
    

# assignment3
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
