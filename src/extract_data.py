## Extract training data (take only data after day X (e.g. ,day 4)
import sys
from dataio import load_data
import argparse

def main():
    parser = argparse.ArgumentParser(description='Duolingo shared task - extract data helper')
    parser.add_argument('data', help='data file name')
    parser.add_argument('--days', help='extract data if day > days', type=float, default=0)

    args = parser.parse_args()

    train = True if "train" in args.data else False
    training_data, _ = load_data(args.data, train=train)
    for data_instance in training_data:
        if data_instance.days > args.days:
            print(data_instance)


if __name__=="__main__":
    main()