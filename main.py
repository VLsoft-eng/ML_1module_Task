import argparse

from src.model import VlClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Spaceship Titanic Classifier')
    parser.add_argument('command', choices=['train', 'predict'], help='Команда для выполнения')
    parser.add_argument('--dataset', required=True, help='Путь к CSV файлу')
    parser.add_argument('--n_trials', type=int, default=20, help='Количество попыток оптимизации')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = VlClassifier()

    if args.command == 'train':
        model.train(args.dataset, args.n_trials)
    elif args.command == 'predict':
        model.predict(args.dataset)