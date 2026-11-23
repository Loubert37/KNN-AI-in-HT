# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Models import CNN, AdamOptimizer, CategoricalCrossEntropy, Trainer

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CSV_PATHS = [
    r"C:\Users\louis\Desktop\AI in HT\Project\mass_case_description_train_set.csv",
    r"C:\Users\louis\Desktop\AI in HT\Project\calc_case_description_train_set.csv",
]
DICOM_ROOT_DIR = r"C:\Users\louis\Desktop\AI in HT\Project\cbis_ddsm"

IMG_SIZE = 96
SAMPLES_PER_CLASS = 1000
CACHE_FILE = f"cbis_cache_{IMG_SIZE}.npz"


# ==============================================================================
# 2. CHARGEMENT AVEC CACHE DISQUE + CLAHE + GROUPEMENT PAR PATIENT
# ==============================================================================
def load_balanced_cbis(csv_paths, dicom_root, img_size=96, samples_per_class=1000):
    if os.path.exists(CACHE_FILE):
        print(f"[+] Chargement instantané depuis le cache '{CACHE_FILE}'...")
        data = np.load(CACHE_FILE)
        return data['X'], data['y'], data['y_int'], data['groups']

    dfs = [pd.read_csv(p) for p in csv_paths if os.path.exists(p)]
    df = pd.concat(dfs, ignore_index=True)

    folder_to_info = {}
    for _, row in df.iterrows():
        top_folder = str(row['cropped image file path']).strip().split('/')[0].strip()
        label = 1 if 'MALIGNANT' in str(row['pathology']).upper() else 0
        pid = str(row['patient_id']).strip()
        folder_to_info[top_folder] = (label, pid)

    print(f"Scan équilibré dans : {dicom_root} ...")
    X_benign, groups_benign = [], []
    X_malignant, groups_malignant = [], []

    folders = list(os.listdir(dicom_root))
    np.random.seed(42)
    np.random.shuffle(folders)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for folder_name in folders:
        if folder_name not in folder_to_info:
            continue

        label, pid = folder_to_info[folder_name]
        if label == 0 and len(X_benign) >= samples_per_class:
            continue
        if label == 1 and len(X_malignant) >= samples_per_class:
            continue

        folder_path = os.path.join(dicom_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith('.dcm'):
                    continue
                try:
                    ds = pydicom.dcmread(os.path.join(root, file))
                    img = ds.pixel_array.astype(np.float32)

                    if len(np.unique(img)) <= 2:
                        continue

                    img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    img_clahe = clahe.apply(img_8u).astype(np.float32)

                    mean, std = img_clahe.mean(), img_clahe.std()
                    if std > 1e-6:
                        img_norm = (img_clahe - mean) / std
                    else:
                        continue

                    img_resized = cv2.resize(img_norm, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    if label == 0:
                        X_benign.append(img_resized)
                        groups_benign.append(pid)
                    else:
                        X_malignant.append(img_resized)
                        groups_malignant.append(pid)
                    break
                except Exception:
                    continue

        if len(X_benign) >= samples_per_class and len(X_malignant) >= samples_per_class:
            break

    min_count = min(len(X_benign), len(X_malignant))
    X_list = X_benign[:min_count] + X_malignant[:min_count]
    y_list = [0] * min_count + [1] * min_count
    groups_list = groups_benign[:min_count] + groups_malignant[:min_count]

    X = np.array(X_list, dtype=np.float32)[..., np.newaxis]
    y_int = np.array(y_list, dtype=np.int64)
    y_onehot = np.eye(2, dtype=np.float32)[y_int]
    groups = np.array(groups_list)

    np.savez_compressed(CACHE_FILE, X=X, y=y_onehot, y_int=y_int, groups=groups)
    print(f"[+] Cache sauvegardé dans '{CACHE_FILE}'.")

    return X, y_onehot, y_int, groups


# ==============================================================================
# 3. ANALYSE DÉTAILLÉE (MATRICE DE CONFUSION, FP/FN, COURBE ROC)
# ==============================================================================
def evaluate_detailed_metrics(model, X_val, y_val, batch_size=64):
    model.set_training(False)

    # Prédiction par batch pour économiser la mémoire
    probs_list = []
    for start_idx in range(0, len(X_val), batch_size):
        end_idx = min(start_idx + batch_size, len(X_val))
        probs_list.append(model.forward(X_val[start_idx:end_idx]))

    y_probs = np.vstack(probs_list)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)
    malin_scores = y_probs[:, 1]  # Probabilité prédite d'être Malin (classe 1)

    # Extraction des 4 cadrans de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    bonnes_identifications = tp + tn
    mauvaises_identifications = fp + fn

    sensibilite = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Taux de cancers détectés (Recall Malin)
    specificite = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Taux de bénins bien écartés
    precision_malin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    auc_score = roc_auc_score(y_true, malin_scores)

    print("\n" + "=" * 65)
    print("        BILAN MÉDICAL DÉTAILLÉ SUR LE JEU DE VALIDATION")
    print("=" * 65)
    print(f"Total d'images évaluées          : {total}")
    print(f"Bonnes identifications totales   : {bonnes_identifications} / {total} ({bonnes_identifications / total * 100:.2f}%)")
    print(f"Mauvaises identifications totales: {mauvaises_identifications} / {total} ({mauvaises_identifications / total * 100:.2f}%)")
    print("-" * 65)
    print(f"  [VN] Vrais Négatifs (Bénin bien identifié)   : {tn}")
    print(f"  [VP] Vrais Positifs (Malin bien identifié)   : {tp}")
    print(f"  [FP] Faux Positifs  (Bénin classé Malin)     : {fp}  <-- Fausse alerte")
    print(f"  [FN] Faux Négatifs  (Malin classé Bénin)     : {fn}  <-- Tumeur manquée")
    print("-" * 65)
    print(f"Sensibilité (Détection des cancers / Recall)   : {sensibilite * 100:.2f}%")
    print(f"Spécificité (Identification des cas bénins)    : {specificite * 100:.2f}%")
    print(f"Précision prédictive positive (Malin)          : {precision_malin * 100:.2f}%")
    print(f"Score ROC-AUC (0.5 = hasard, 1.0 = parfait)    : {auc_score:.4f}")
    print("=" * 65)

    print("\nRapport complet Scikit-Learn :")
    print(classification_report(y_true, y_pred, target_names=["Bénin (0)", "Malin (1)"], zero_division=0))

    # --- Affichage graphique : Matrice de confusion + Courbe ROC ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Matrice de confusion graphique
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].figure.colorbar(im, ax=axes[0])
    classes = ['Bénin (0)', 'Malin (1)']
    axes[0].set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        title='Matrice de Confusion (Validation)',
        ylabel='Diagnostic Réel',
        xlabel='Prédiction du CNN'
    )

    labels_grid = [
        [f"VN: {tn}\n({tn/total*100:.1f}%)", f"FP: {fp}\n({fp/total*100:.1f}%)"],
        [f"FN: {fn}\n({fn/total*100:.1f}%)", f"VP: {tp}\n({tp/total*100:.1f}%)"]
    ]
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j, i, labels_grid[i][j],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    # 2. Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, malin_scores)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {auc_score:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Hasard (AUC = 0.500)')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taux de Faux Positifs (1 - Spécificité)')
    axes[1].set_ylabel('Taux de Vrais Positifs (Sensibilité)')
    axes[1].set_title('Courbe ROC (Capacité de discrimination)')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. ENTRAÎNEMENT DU CNN OPTIMISÉ
# ==============================================================================
def main():
    X, y, y_int, groups = load_balanced_cbis(CSV_PATHS, DICOM_ROOT_DIR, IMG_SIZE, SAMPLES_PER_CLASS)

    print(f"\n[+] Dataset équilibré prêt : {len(X)} images ({IMG_SIZE}x{IMG_SIZE})")
    print(f"    Répartition -> Bénin (0) : {np.sum(y_int == 0)} | Malin (1) : {np.sum(y_int == 1)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y_int, groups=groups))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    architecture = [
        ("conv", 1, 32, 3, 1, True),
        ("bn", 32),
        ("activation", "leaky_relu"),
        ("conv", 32, 64, 3, 1, True),
        ("bn", 64),
        ("activation", "leaky_relu"),
        ("conv", 64, 128, 3, 1, True),
        ("bn", 128),
        ("activation", "leaky_relu"),
        ("conv", 128, 256, 3, 1, False),
        ("bn", 256),
        ("activation", "leaky_relu"),
        ("gap",),
        ("dropout", 0.3),
        ("dense", 256, 64),
        ("activation", "leaky_relu"),
        ("dense", 64, 2),
        ("activation", "softmax")
    ]

    model = CNN(architecture)
    criterion = CategoricalCrossEntropy()
    optimizer = AdamOptimizer(model, lr=0.001, weight_decay=1e-4)
    trainer = Trainer(model, criterion, optimizer)

    print("\nDémarrage de l'entraînement (BatchNorm + AdamW + CLAHE + Data Augmentation)...\n")
    trainer.fit(
        X_train, y_train,
        epochs=35,
        batch_size=32,
        X_val=X_val,
        y_val=y_val,
        augment=True,
        lr_decay_step=12
    )

    model.load_weights("best_model_weights.npz")

    # Affichage des courbes d'entraînement + bilan médical complet
    trainer.plot_metrics()
    evaluate_detailed_metrics(model, X_val, y_val)


if __name__ == "__main__":
    main()