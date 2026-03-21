class Vehicule:
    nombre_vehicules = 0

    def __init__(self, annee, prix):
        Vehicule.nombre_vehicules += 1
        self.matricule = Vehicule.nombre_vehicules
        self.__annee = annee
        self.__prix = prix

    def getAnnee(self):
        return self.__annee

    def getPrix(self):
        return self.__prix

    def demarrer(self):
        print("Le véhicule démarre")

    def accelerer(self):
        print("Le véhicule accélère")

    def ToString(self):
        return f"Matricule: {self.matricule}, Année: {self.__annee}, Prix: {self.__prix}"

class Voiture(Vehicule):
    def demarrer(self):
        print("La voiture démarre silencieusement")
    
    def accelerer(self):
        print("La voiture accélère rapidement")

class Camion(Vehicule):
    def demarrer(self):
        print("Le camion démarre bruyamment")
    
    def accelerer(self):
        print("Le camion accélère lentement")

v1 = Voiture(2020, 15000)
c1 = Camion(2018, 45000)
v2 = Voiture(2022, 20000)

print(v1.ToString())
v1.demarrer()
print(c1.ToString())
c1.accelerer()
print(v2.ToString()) 