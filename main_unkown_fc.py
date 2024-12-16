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

target_x = np.linspace(0, 1, 100)
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
target_y = np.array([-1.36936457e-07, -1.39334916e-07, -1.41733374e-07, -1.44131833e-07,
       -1.46530291e-07, -1.48928750e-07, -1.51327209e-07, -1.53725667e-07,
       -1.56124126e-07, -1.58522584e-07, -1.60921043e-07, -1.63319502e-07,
       -1.65717960e-07, -1.68116419e-07, -1.70514878e-07, -1.72913336e-07,
       -1.75311795e-07, -1.77710253e-07, -1.80108712e-07, -1.82507171e-07,
       -1.84905629e-07, -1.87304088e-07, -1.89702547e-07, -1.92101005e-07,
       -1.94499464e-07,  5.03073733e-05,  1.21386139e-03,  3.78767317e-03,
        7.62063308e-03,  1.25825089e-02,  1.85603763e-02,  2.54559479e-02,
        3.31831607e-02,  4.16664289e-02,  5.08390100e-02,  6.06418324e-02,
        7.10224764e-02,  8.19339388e-02,  9.33343658e-02,  1.05186166e-01,
        1.17455538e-01,  1.30111812e-01,  1.43127406e-01,  1.56477232e-01,
        1.70138504e-01,  1.84090631e-01,  1.98314345e-01,  2.12792516e-01,
        2.27509171e-01,  2.42449624e-01,  2.57600713e-01,  2.72949732e-01,
        2.88485482e-01,  3.04197328e-01,  3.20075384e-01,  3.36110760e-01,
        3.52294950e-01,  3.68620053e-01,  3.85078904e-01,  4.01664617e-01,
        4.18370820e-01,  4.35191741e-01,  4.52121624e-01,  4.69155484e-01,
        4.86288302e-01,  5.03515514e-01,  5.20832973e-01,  5.38236481e-01,
        5.55722280e-01,  5.73286840e-01,  5.90926521e-01,  6.08638763e-01,
        6.26420068e-01,  6.44267630e-01,  6.62178825e-01,  6.80151108e-01,
        6.98182060e-01,  7.16269323e-01,  7.34411009e-01,  7.52604654e-01,
        7.70848671e-01,  7.89140970e-01,  8.07479795e-01,  8.25863601e-01,
        8.44290666e-01,  8.62759294e-01,  8.81268635e-01,  8.99816684e-01,
        9.18402294e-01,  9.37024290e-01,  9.55681295e-01,  9.74372140e-01,
        9.93096265e-01,  1.01185160e+00,  1.03063836e+00,  1.04945448e+00,
        1.06829965e+00,  1.08717274e+00,  1.10607275e+00,  1.12499938e+00])
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
