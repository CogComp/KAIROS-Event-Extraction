# This is the main file
from util import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='1', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--representation_source", default='nyt', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--model", default='bert-large', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--pooling_method", default='final', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--weight", default=100, type=float, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--argument_matching", default='exact', type=str, required=False,
                    help="weight assigned to triggers")
parser.add_argument("--eval_model", default='joint', type=str, required=False,
                    help="weight assigned to triggers")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current device:', device)


test_extractor = CogcompKairosEventExtractor(device)

test_results = test_extractor.extract('Bob attacks Kevin.')

print(test_results)

test_results = test_extractor.extract('Bob hits Kevin.')
print(test_results)
print('end')
