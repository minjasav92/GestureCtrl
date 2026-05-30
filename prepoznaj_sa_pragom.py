def prepoznaj_sa_pragom(vektor, model, prag=0.6):
    proba = model.predict_proba(vektor.reshape(1, -1))[0]
    if proba.max() < prag:
        return None
    indeks = proba.argmax()
    return model.classes_[indeks]