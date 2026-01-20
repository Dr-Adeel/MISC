# Exercice 1
# Q1
from abc import ABC, abstractmethod
class Vehicule(ABC):
    __matricule = 0
    def __init__(self,annee,prix):
        Vehicule.__matricule += 1
        self.__id = Vehicule.__matricule
        self.__annee=annee
        self.__prix=prix

    def get_matricule(self):
        return self.__id
    def get_annee(self):
        return self.__annee
    def get_prix(self):
        return self.__prix
    def set_prix(self,prix):
        self.__prix=prix
    def set_annee(self,annee):
        self.__annee=annee
    def __str__(self):
        return f"Matricule: {self.__id}, Annee: {self.__annee}, Prix: {self.__prix}"
    @abstractmethod
    def demarrer(self):
        pass
    @abstractmethod
    def accelerer(self):
        pass
# Q2
class Voiture(Vehicule):
    def demarrer(self):
        return "La voiture demarre"
    def accelerer(self):
        return "La voiture accelere"
    
class Camion(Vehicule):
    def demarrer(self):
        return "Le camion demarre"
    def accelerer(self):
        return "Le camion accelere"
# Q3
class Test:
    def __init__(self):
        v1=Voiture(2020,20000)
        print(v1)
        print(v1.demarrer())
        print(v1.accelerer())
        c1=Camion(2018,50000)
        print(c1)
        print(c1.demarrer())
        print(c1.accelerer())
print("----Test Vehicule----")
Test()


#Exercice 2
# Q1
class Batiment:
    def __init__(self,adresse= None):
        self.__adresse=adresse


    def get_adresse(self):
        return self.__adresse
    
    def set_adresse(self,adresse):
        self.__adresse=adresse

    def __str__(self):
        return f"Adresse: {self.__adresse}"
    
# Q2
class Maison(Batiment):
    def __init__(self,adresse=None,NbPieces= 0):
        super().__init__(adresse)
        self.__NbPieces=NbPieces

    def get_NbPieces(self):
        return self.__NbPieces
    
    def set_adresse(self, adresse):
        return super().set_adresse(adresse)
    def set_NbPieces(self,NbPieces):
        self.__NbPieces=NbPieces
    def get_adresse(self):
        return super().get_adresse()
    def __str__(self):
        return f"{super().__str__()}, Nombre de pieces: {self.__NbPieces}"

# Q3
class TestBatiment:
    def __init__(self):
        b1=Batiment("123 Rue de Paris")
        print(b1.__str__())
        m1=Maison("456 Avenue de Lyon",5)
        print(m1.__str__())
print("----Test Batiment----")
TestBatiment()

# Exercise 3
# Q1
class Personne:
    def __init__(self,nom,prenom,date_naissance):
        self.__nom=nom
        self.__prenom=prenom
        self.__date_naissance=date_naissance

    def Afficher(self):
        return f"Nom: {self.__nom}, Prenom: {self.__prenom}, Date de naissance: {self.__date_naissance}"
    
class Employe(Personne):
    def __init__(self,nom,prenom,date_naissance,salaire):
        super().__init__(nom,prenom,date_naissance)
        self.__salaire=salaire

    def Afficher(self):
        return f"{super().Afficher()}, Salaire: {self.__salaire}"
    

class Chef(Employe):
    def __init__(self,nom,prenom,date_naissance,salaire,service):
        super().__init__(nom,prenom,date_naissance,salaire)
        self.__service=service

    def Afficher(self):
        return f"{super().Afficher()}, Service: {self.__service}"
    
class Directeur(Chef):
    def __init__(self,nom,prenom,date_naissance,salaire,service,departement):
        super().__init__(nom,prenom,date_naissance,salaire,service)
        self.__departement=departement

    def Afficher(self):
        return f"{super().Afficher()}, Departement: {self.__departement}"

# Q2
class TestPersonne:
    Personnes = []
    def Creer_Personnes(self):
        self.Personnes.append(Employe("Doe","John","01/01/1990",3000))
        self.Personnes.append(Employe("Smith","Jane","02/02/1985",3500))
        self.Personnes.append(Employe("Brown","Mike","03/03/1975",4000))
        self.Personnes.append(Employe("Johnson","Emily","04/04/1980",5000))
        self.Personnes.append(Employe("Davis","Chris","05/05/1970",6000))
        self.Personnes.append(Chef("Davis","Chris","05/05/1970",6000,"IT"))
        self.Personnes.append(Chef("Miller","Sarah","07/07/1988",7000,"Finance"))
        self.Personnes.append(Directeur("Wilson","Anna","06/06/1965",8000,"HR","North America"))
        return self.Personnes
    
# Q3
print("----Test Personne----")
Test=TestPersonne()
for personne in Test.Creer_Personnes():
    print(personne.Afficher())


# Exercice 4 
# Q1
class Employe:
    def __init__(self,Matricule,Nom,Prenom,Date_naissance):
        self.__Matricule=Matricule
        self.__Nom=Nom
        self.__Prenom=Prenom
        self.__Date_naissance=Date_naissance

    def get_Matricule(self):
        return self.__Matricule
    def get_Nom(self):
        return self.__Nom
    def get_Prenom(self):
        return self.__Prenom
    def get_Date_naissance(self):
        return self.__Date_naissance
    def set_Nom(self,Nom):
        self.__Nom=Nom
    def set_Prenom(self,Prenom):
        self.__Prenom=Prenom
    def set_Date_naissance(self,Date_naissance):
        self.__Date_naissance=Date_naissance
    def __str__(self):
        return f"Matricule: {self.__Matricule}, Nom: {self.__Nom}, Prenom: {self.__Prenom}, Date de naissance: {self.__Date_naissance}"
    
    def GetSalaire(self):
        pass

class Ouvrier(Employe):
    SMIG = 2500
    def __init__(self, Matricule, Nom, Prenom, Date_naissance,date_entree):
        super().__init__(Matricule, Nom, Prenom, Date_naissance)
        self.__datee_entree = date_entree

    def __str__(self):
        return super().__str__() + f", Date d'entree: {self.__datee_entree}"

    @property
    def anciennete(self):
        from datetime import datetime
        current_year = datetime.now().year
        entry_year = int(self.__datee_entree.split('/')[-1])
        return current_year - entry_year
    def GetSalaire(self):
        salaire_ouvrier = Ouvrier.SMIG + (self.anciennete * 100)
        if salaire_ouvrier < Ouvrier.SMIG*2:
            return salaire_ouvrier
        else:
            return Ouvrier.SMIG*2
    

    

class Cadre(Employe):
    def __init__(self, Matricule, Nom, Prenom, Date_naissance,indice):
        super().__init__(Matricule, Nom, Prenom, Date_naissance)
        self.__indice = indice

    def __str__(self):
        return super().__str__() + f", Indice: {self.__indice}"
    def GetSalaire(self):
        match self.__indice:
            case 1:
                return 13000
            case 2:
                return 15000
            case 3:
                return 17000
            case 4:
                return 20000

class Patron(Employe):
    Chiffre_Affaires = 1000000
    def __init__(self, Matricule, Nom, Prenom, Date_naissance,pourcentage_societe):
        super().__init__(Matricule, Nom, Prenom, Date_naissance)
        self.__pourcentage_societe = pourcentage_societe

    def __str__(self):
        return super().__str__() + f", Pourcentage societe: {self.__pourcentage_societe}"
    def GetSalaire(self):
        return Patron.Chiffre_Affaires * self.__pourcentage_societe / 100
    
class TestEmploye:
    def __init__(self):
        e1=Ouvrier(1,"Doe","John","01/01/1990","15/06/2015")
        print(e1.__str__())
        print(f"Salaire: {e1.GetSalaire()}")
        e2=Cadre(2,"Smith","Jane","02/02/1985",3)
        print(e2.__str__())
        print(f"Salaire: {e2.GetSalaire()}")
        e3=Patron(3,"Brown","Mike","03/03/1975",10)
        print(e3.__str__())
        print(f"Salaire: {e3.GetSalaire()}")

print("----Test Employe----")
TestEmploye()