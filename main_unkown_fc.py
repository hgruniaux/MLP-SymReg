from symreg.formula import *
from symreg.generator import *
from symreg.mutator import *
#from symreg.crossover import *
from symreg.random import *
from symreg.evolution import *
from symreg.tokenizer import tokenize

import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * INPUT_SAMPLES, 128)  # 100 is the dimension after conv layers
        self.fc2 = nn.Linear(128, OUTPUT_FEATURES)
        
    def forward(self, x):
        # Input shape: [batch_size, 100, 2]
        
        # Transpose to [batch_size, 2, 100] for Conv1D (since PyTorch expects [batch_size, channels, length])
        x = x.transpose(1, 2)
        
        # Convolutional layers + activation
        x = F.relu(self.conv1(x))
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Fully connected layers + activation
        x = F.relu(self.fc1(x))
        
        # Output layer
        x = self.fc2(x)
        
        return x

mpl.use("QtAgg")

np.seterr(all="ignore")

parser = argparse.ArgumentParser(
    prog="SymReg Driver",
    description="Test the evolutionary algorithm for symbolic regression",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Show intermediate information"
)
parser.add_argument(
    "-n", "--noise", action="store_true", help="Add noise to the target function"
)
parser.add_argument(
    "-o", "--optimizer", action="store_true", help="Enable constant optimizer mutation"
)
parser.add_argument(
    "-i", "--iterations", type=int, default=10, help="Number of iterations"
)
parser.add_argument("-p", "--population", type=int, default=500, help="Population size")
parser.add_argument(
    "-k", "--kbest", type=int, default=5, help="Number of best candidates"
)
parser.add_argument("-c", "--crossover", type=float, default=0.0, help="Crossover rate")
parser.add_argument("-m", "--mutation", type=float, default=0.95, help="Mutation rate")
parser.add_argument(
    "-d",
    "--maxdepth",
    type=int,
    default=3,
    help="Maximum depth for expression generation",
)
parser.add_argument(
    "-l", "--lambda_coef", type=float, default=0.1, help="Regularization parameter"
)
args = parser.parse_args()

target_x = np.array([ 2.048,  2.176,  2.304,  2.432,  2.56 ,  2.688,  2.816,  2.944,
        3.072,  3.2  ,  3.328,  3.456,  3.584,  3.712,  3.84 ,  3.968,
        4.096,  4.224,  4.352,  4.48 ,  4.608,  4.736,  4.864,  4.992,
        5.12 ,  5.248,  5.376,  5.504,  5.632,  5.76 ,  5.888,  6.016,
        6.144,  6.272,  6.4  ,  6.528,  6.656,  6.784,  6.912,  7.04 ,
        7.168,  7.296,  7.424,  7.552,  7.68 ,  7.808,  7.936,  8.064,
        8.192,  8.32 ,  8.448,  8.576,  8.704,  8.832,  8.96 ,  9.088,
        9.216,  9.344,  9.472,  9.6  ,  9.728,  9.856,  9.984, 10.112,
       10.24 , 10.368, 10.496, 10.624, 10.752, 10.88 , 11.008, 11.136,
       11.264, 11.392])
random_options = RandomOptions()
random_options.definition_set = target_x
random_options.max_depth = args.maxdepth



#target_formula = random_formula(random_options)
#target_formula = UnaryExpression(UnaryOp.SIN, VariableExpression() * 4.3)
# target_formula = VariableExpression() * 4.3
x = VariableExpression(0)
#target_formula = UnaryExpression(UnaryOp.SIN, UnaryExpression(UnaryOp.SQRT, x*4)) + 1.2


#print(f"Target: {target_formula}")
#print(f"Tokens: {tokenize(target_formula)}")
#target_y = target_formula(target_x)
target_y = np.array([622., 563., 541., 533., 521., 406., 431., 401., 314., 312., 289., 306., 267., 281., 268., 282., 248., 270., 195., 194., 225., 209.,
       166., 163., 168., 117., 149., 143., 113.,  84., 127.,  80., 111.,
       118.,  75.,  90.,  84.,  70.,  80.,  69.,  93.,  45.,  71.,  69.,
        69.,  57.,  86.,  43.,  52.,   6.,  30.,   6.,  26.,  -8., -13.,
        32.,  33.,  23.,  26., -14.,   3.,  17.,   2.,  21.,  10.,  -6.,
        23., -11., -14.,  19.,  -4., -18., -16.,  -6.])
print(f"Mean = {np.mean(target_y)}, Std = {np.std(target_y)}")
if args.noise:
    target_y += 0.1 * np.random.normal(
        np.mean(target_y), np.std(target_y), np.shape(target_y)
    )

model = torch.load("CNN.model")

def normalize(X, Y, size: int = 100):
    """
    Z-score normalization (standardization).
    """

    idx = np.round(np.linspace(0, len(X) - 1, size)).astype(int)
    X_norm = X[idx]
    Y_norm = ((Y - np.mean(Y)) / np.std(Y))[idx]
    return X_norm, Y_norm

def predict(target_x, target_y):
    X_norm, Y_norm = normalize(target_x, target_y, 100)
    Y_norm = torch.tensor(np.array([X_norm,Y_norm]).T, dtype=torch.float32).unsqueeze(0)
    preds = model(Y_norm)
    return preds

proba_unary = F.softmax(predict(target_x, target_y))
print(f"Unary probability from CNN : {proba_unary}")  

print(proba_unary.flatten().shape)
random_options.prob_unary_operators = proba_unary.flatten().detach().numpy()

generator = RandomGenerator(random_options)
mutator = SequentialMutator(
    [
        RandomMutator(
            [
                ConstantNoiseMutator(),
                SwapBinaryExpressionMutator(),
                BinaryOperatorMutator(),
                BinaryInserterMutator(),
                BinaryRemoverMutator(),
                UnaryOperatorMutator(),
                UnaryInserterMutator(),
                UnaryRemoverMutator(),
            ]
            + ([ConstantOptimizerMutator(target_x, target_y)] if args.optimizer else [])
        ),
        SimplifyMutator(),
    ]
)

options = Options()
options.verbose = args.verbose
options.iterations = args.iterations
options.population_size = args.population
options.crossover_rate = args.crossover
options.mutation_rate = args.mutation
options.lambda_regularization = args.lambda_coef
options.k_best = args.kbest
options.generator = generator
options.mutator = mutator

plt.ion()

fig = plt.figure()
ax = plt.subplot(111)

ax.set_title("Symbolic Regression")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(
    target_x,
    target_y,
    linestyle="dashed",
)

lines = [
    ax.plot(target_x, np.zeros_like(target_x))[0] for i in range(options.k_best)
 ]

labels = [f"Target function: $?$"] + [f"Candidate {i}" for i in range(options.k_best)]

ax.legend(labels=labels)

def updater(candidates):
    for i, candidate in enumerate(candidates):
        y = candidate(target_x)
        if np.shape(y) != np.shape(target_x):
            y = np.array([y] * len(target_x))
        labels[i + 1] = f"Candidate {i}: ${candidate.latex()}$"
        lines[i].set_ydata(y)

    ax.legend(labels=labels)
    fig.canvas.draw()
    fig.canvas.flush_events()

options.updater = updater
candidates = run(target_x, target_y, options)

print("Finished, there is the best candidates:")
for i, candidate in enumerate(candidates):
    print(f"  - Candidate {i}: {candidate}")
plt.show(block=True)
# plt.savefig("output.png")
