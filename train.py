from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

dados = pd.read_csv('Dados\dado_fake.csv.csv')

x = dados.iloc[:, :-1]
y = dados.iloc[:, -1]

x_treino, x_teste, y_treino, y_teste = train_test_split(x,
                                                        y,
                                                        stratify = y,
                                                        random_state = 10)

rf = RandomForestClassifier(random_state = 12)

rf.fit(x_treino, y_treino)

joblib.dump(rf, 'model.pkl')