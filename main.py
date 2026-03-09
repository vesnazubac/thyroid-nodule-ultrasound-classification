import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np
import random

from src.dataset import ThyroidDataset   # pretpostavljam da već postoji
from src.model import (
    BaselineCNN,
    DenseNet121Transfer,
    ResNet50Transfer,
    EfficientNetB0Transfer,
    CheXNetDenseNet121
)
from src.utils import get_transforms
from src.train import train_model


def set_seed(seed=42):
    """Postavlja seme za reproduktivnost."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Thyroid Nodule Classification')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'densenet', 'resnet', 'efficientnet', 'chexnet'],
                        help='Arhitektura modela')
    parser.add_argument('--epochs', type=int, default=10, help='Broj epoha')
    parser.add_argument('--lr', type=float, default=1e-4, help='Početna stopa učenja')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'],
                        help='Vrsta loss funkcije')
    parser.add_argument('--batch_size', type=int, default=16, help='Veličina batch-a')
    parser.add_argument('--seed', type=int, default=42, help='Seed za reproduktivnost')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Koristi težine klasa za CrossEntropy')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Podešavanje ulazne veličine u zavisnosti od modela
    if args.model == 'baseline':
        input_size = 128
    else:
        input_size = 224   # svi transfer modeli očekuju 224

    # Transformacije
    train_transform, val_transform = get_transforms(input_size=input_size)

    # Učitavanje celog dataseta (sa trening transformacijama za sada, ali kasnije ćemo preslikati)
    # Pretpostavka: dataset klasa prima transform parametar
    image_dir = "data/TN5000_forReview/JPEGImages"
    annotation_dir = "data/TN5000_forReview/Annotations"

    full_dataset = ThyroidDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=train_transform   # privremeno, ali ćemo za validaciju koristiti val_transform
    )

    # Prikupljanje labela za stratifikovanu podelu
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    # Prvo podeli na trening i privremeni (validacija+test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
    train_idx, temp_idx = next(sss.split(np.zeros(len(labels)), labels))

    # Zatim podeli privremeni na validaciju i test (50% od 30% = 15% svaki)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=args.seed
    )

    # Kreiraj podskupove sa odgovarajućim transformacijama
    train_dataset = Subset(full_dataset, train_idx)
    # Za validaciju i test koristimo val_transform (bez augmentacije)
    # Najlakše: ponovo kreiramo dataset sa val_transform, ali samo za te indekse
    # Alternativa: dinamički menjati transform, ali ovako je jednostavnije:
    val_dataset = ThyroidDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=val_transform
    )
    test_dataset = ThyroidDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=val_transform
    )
    # Sada ograničavamo na odgovarajuće indekse
    val_dataset = Subset(val_dataset, val_idx)
    test_dataset = Subset(test_dataset, test_idx)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Odabir modela
    if args.model == 'baseline':
        model = BaselineCNN()
    elif args.model == 'densenet':
        model = DenseNet121Transfer(num_classes=2, pretrained=True)
    elif args.model == 'resnet':
        model = ResNet50Transfer(num_classes=2, pretrained=True)
    elif args.model == 'efficientnet':
        model = EfficientNetB0Transfer(num_classes=2, pretrained=True)
    elif args.model == 'chexnet':
        model = CheXNetDenseNet121(num_classes=2)
    else:
        raise ValueError("Nepoznat model")

    print(f"Model: {args.model}, ukupno parametara: {sum(p.numel() for p in model.parameters())}")

    # Treniranje
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        use_class_weights=args.use_class_weights,
        loss_type=args.loss
    )

    # Nakon treninga, učitaj najbolji model i testiraj
    model.load_state_dict(torch.load('best_model.pth'))
    print("\nEvaluacija na test skupu:")
    from src.train import validate
    precision, recall, f1, roc_auc, _ = validate(model, test_loader, device)
    print(f"Test -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()