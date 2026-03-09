import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    """
    Focal Loss za nebalansirane klase.
    alpha: težinski faktor za klase (obično 0.25 za pozitivnu klasu)
    gamma: faktor fokusiranja (2 radi dobro)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4,
                use_class_weights=False, loss_type='ce', patience=5):
    """
    Trenira model i prati metrike na validacionom skupu.
    - use_class_weights: da li da koristi težine klasa za CrossEntropy
    - loss_type: 'ce' (CrossEntropy) ili 'focal' (Focal Loss)
    - patience: rano zaustavljanje ako nema poboljšanja F1 kroz ovoliko epoha
    """
    model.to(device)

    # Odabir loss funkcije
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        if use_class_weights:
            # Težine se računaju na trening skupu (pretpostavljamo da su labele dostupne)
            train_labels = []
            for _, lbl in train_loader.dataset:
                train_labels.append(lbl)
            class_counts = torch.bincount(torch.tensor(train_labels))
            class_weights = 1. / class_counts.float()
            class_weights = class_weights / class_weights.sum()  # normalizacija
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

    # Praćenje metrika
    precision_list, recall_list, f1_list, roc_auc_list = [], [], [], []
    train_loss_list, val_loss_list = [], []

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # ====== TRENING ======
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # ====== VALIDACIJA ======
        precision, recall, f1, roc_auc, val_loss = validate(model, val_loader, device, criterion)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)
        val_loss_list.append(val_loss)

        # Scheduler prati ROC-AUC (možeš i F1)
        scheduler.step(roc_auc)

        # Rano zaustavljanje na osnovu F1
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("✅ Model sačuvan (najbolji F1).")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Rano zaustavljanje nakon {epoch+1} epoha.")
                break

    # ====== GRAFICI ======
    epochs_range = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, precision_list, label='Precision')
    plt.plot(epochs_range, recall_list, label='Recall')
    plt.plot(epochs_range, f1_list, label='F1-score')
    plt.plot(epochs_range, roc_auc_list, label='ROC-AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def validate(model, loader, device, criterion=None):
    """
    Evaluacija modela na datom loaderu.
    Vraća precision, recall, f1, roc_auc i prosečan loss (ako je criterion dat).
    """
    model.eval()
    all_labels = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # vjerovatnoće za klasu 1

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)

    avg_loss = total_loss / len(loader) if criterion is not None else 0.0

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    if criterion is not None:
        print(f"Val Loss: {avg_loss:.4f}")

    return precision, recall, f1, roc_auc, avg_loss