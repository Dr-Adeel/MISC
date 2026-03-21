class Batiment:
    def __init__(self, adresse="Inconnue"):
        self.adresse = adresse

    def setAdresse(self, adr):
        self.adresse = adr

    def getAdresse(self):
        return self.adresse

    def ToString(self):
        return f"Bâtiment situé à : {self.adresse}"

class Maison(Batiment):
    def __init__(self, adresse, nbPieces):
        super().__init__(adresse)
        self.nbPieces = nbPieces

    def setNbPieces(self, nb):
        self.nbPieces = nb

    def getNbPieces(self):
        return self.nbPieces

    def ToString(self):
        return super().ToString() + f", Nombre de pièces : {self.nbPieces}"

bat = Batiment("12 rue de la Paix")
maison = Maison("5 avenue des Champs", 4)

print(bat.ToString())
print(maison.ToString())