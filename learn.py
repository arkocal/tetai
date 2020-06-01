import argparse
import random

from mechanics import Mechanics
import utils

parser = argparse.ArgumentParser(description='Learn from dumped games')
parser.add_argument('--ai', dest='ai', choices=["sklearn_mlp", "torch"],
                    required=True,
                    help='AI player that should learn')
parser.add_argument('--ai-data-in', dest='ai_data_in',
                    help='Load pre-trained model parameters, optional')
parser.add_argument('--ai-data-out', dest='ai_data_out', required=True,
                    help='Path to dump parameters after training.')
parser.add_argument('--training-data', dest='training_data', required=True)
parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                    help="Number of iterations.")
parser.add_argument('--batch-size', dest='batch_size', default=-1, type=int,
                    help="Batch size for each epoch, -1 uses all dataset.")
# parser.add_argument('--shuffle-training-data', dest="shuffle_training_data", default=True)

args = parser.parse_args()

tetris = Mechanics()
ai_player = utils.get_ai_player(args.ai, tetris, args.ai_data_in)

print("Created/loaded model")
with open(args.training_data, "r") as training_data_file:
    training_data = [(utils.deserialize_field(line.split()[0]),
                      float(line.split()[1])) for line in training_data_file]
print("Loaded data")

batch_size = args.batch_size
if batch_size == -1:
    batch_size = len(training_data)

# TODO bar
for epoch in range(args.epochs):
    print(epoch)
    batch = random.sample(training_data, batch_size)
    ai_player.train(batch)
print("Done training")
ai_player.dump(args.ai_data_out)
print("Dumped to", args.ai_data_out)
