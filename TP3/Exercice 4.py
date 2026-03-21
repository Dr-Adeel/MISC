from abc import ABC, abstractmethod

class Employe(ABC):
    def __init__(self, matricule, nom, prenom):
        self.matricule = matricule
        self.nom = nom
        self.prenom = prenom

    def ToString(self):
        return f"{self.nom} {self.prenom} (Mat: {self.matricule})"

    @abstractmethod
    def GetSalaire(self):
        pass

class Ouvrier(Employe):
    SMIG = 2500 
    
    def __init__(self, matricule, nom, prenom, date_entree):
        super().__init__(matricule, nom, prenom)
        self.date_entree = date_entree

    def GetSalaire(self):
        anciennete = 2023 - self.date_entree 
        salaire = Ouvrier.SMIG + (anciennete * 100)
        
        if salaire > Ouvrier.SMIG * 2:
            return Ouvrier.SMIG * 2
        return salaire

class Cadre(Employe):
    def __init__(self, matricule, nom, prenom, indice):
        super().__init__(matricule, nom, prenom)
        self.indice = indice

    def GetSalaire(self):
        if self.indice == 1: return 13000
        elif self.indice == 2: return 15000
        elif self.indice == 3: return 17000
        elif self.indice == 4: return 20000
        else: return 0

class Patron(Employe):
    CA = 1000000 
    
    def __init__(self, matricule, nom, prenom, pourcentage):
        super().__init__(matricule, nom, prenom)
        self.pourcentage = pourcentage

    def GetSalaire(self):
        return Patron.CA * self.pourcentage / 100

ouvrier = Ouvrier(1, "Dupont", "Pierre", 2010)
cadre = Cadre(2, "Durand", "Marie", 3)
patron = Patron(3, "Martin", "Luc", 10) # 10% du CA

print(f"{ouvrier.ToString()} - Salaire : {ouvrier.GetSalaire()}")
print(f"{cadre.ToString()} - Salaire : {cadre.GetSalaire()}")
print(f"{patron.ToString()} - Salaire : {patron.GetSalaire()}")