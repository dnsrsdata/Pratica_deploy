from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


dados = pd.read_csv(r'Dados\dado_fake.csv')

x = dados.iloc[:, :-1]
y = dados.iloc[:, -1]
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, stratify=y, random_state=10,
)

rf = RandomForestClassifier(random_state=12)

rf.fit(x_treino, y_treino)

joblib.dump(rf, 'model.pkl')
