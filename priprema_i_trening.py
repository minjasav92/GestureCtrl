import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# =====================================================================
# 01 — INŽENJER PODATAKA
# =====================================================================

# Korak 1 — Čišćenje podataka
df = pd.read_csv('dataset_final.csv')
print("Nedostajuće vrednosti:\n", df.isnull().sum())
print("Duplikati:", df.duplicated().sum())

### START CODE HERE ###
df = df.dropna()
df = df.drop_duplicates()
### END CODE HERE ###
print(f'Preostalo redova: {len(df)}')

# Korak 2 — Normalizacija koordinata
coord_cols = [c for c in df.columns if c not in ['label', 'ucesnik', 'frame']]

# Pravimo rečnik za nove kolone da izbegnemo PerformanceWarning fragmentaciju
nove_kolone = {}
for col in coord_cols:
    axis = col[0]  # Uzima 'x', 'y' ili 'z'
    ### START CODE HERE ###
    referentna = df[axis + '0']
    nove_kolone[col + '_norm'] = df[col] - referentna
    ### END CODE HERE ###

# Dodajemo sve normalizovane kolone odjednom u DataFrame
df = pd.concat([df, pd.DataFrame(nove_kolone, index=df.index)], axis=1)

# Korak 3 — Finalni dataset
### START CODE HERE ###
df.dropna().to_csv('dataset_clean.csv', index=False)
df.to_csv('dataset_normalized.csv', index=False)
### END CODE HERE ###


# =====================================================================
# 02 — ISTRAŽIVAČ KLASIFIKACIJE
# =====================================================================

# Korak 1 — Učitavanje dataseta i podela na trening/test
df_trening = pd.read_csv('dataset_normalized.csv')

### START CODE HERE ###
# Dinamički pronalazimo metapodatke koji stvarno postoje u tvom fajlu
metapodaci = [c for c in ['label', 'ucesnik', 'frame'] if c in df_trening.columns]

# Izbacujemo metapodatke i zadržavamo samo čiste x, y, z kolone (63 komada)
X = df_trening.drop(columns=metapodaci)
X = X[[c for c in X.columns if not c.endswith('_norm')]]
y = df_trening['label']
### END CODE HERE ###

# Podela na trening i test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Trening: {len(X_train)}, Test: {len(X_test)}')

# Korak 2 — Trening KNN
### START CODE HERE ###
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f'KNN tacnost: {accuracy_score(y_test, knn.predict(X_test)):.3f}')
### END CODE HERE ###

# Korak 3 — Hiperparametri — odabir optimalnog k
tacnosti = []
for k in range(1, 21):
    ### START CODE HERE ###
    model_k = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(X_train, y_train)
    tacnosti.append(accuracy_score(y_test, model_k.predict(X_test)))
    ### END CODE HERE ###

plt.figure()
plt.plot(range(1, 21), tacnosti, marker='o')
plt.xlabel('k')
plt.ylabel('Tacnost')
plt.title('KNN — izbor k')
plt.savefig('knn_hiperparametri.png')
plt.close()


# =====================================================================
# 03 — ISTRAŽIVAČ EVALUACIJE
# =====================================================================

# Korak 1 — Confusion matrix
### START CODE HERE ###
ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test, cmap='Blues', xticks_rotation=30)
### END CODE HERE ###
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Korak 2 — Analiza grešaka
### START CODE HERE ###
y_pred = knn.predict(X_test)
maska = y_pred != y_test
pogresni = X_test[maska].copy()
pogresni['stvarno'] = y_test[maska].values
pogresni['predvidjeno'] = y_pred[maska]
print("\nMatrica pogrešnih klasifikacija:")
if len(pogresni) > 0:
    print(pogresni[['stvarno', 'predvidjeno']].value_counts())
else:
    print("Nema pogrešnih klasifikacija!")
### END CODE HERE ###

# Korak 3 — Čuvanje modela
### START CODE HERE ###
with open('model.pkl', 'wb') as f:
    pickle.dump(knn, f)
### END CODE HERE ###

print("\n[INFO] Model i grafikoni su uspešno generisani i sačuvani!")