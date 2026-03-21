class Personne:
    def __init__(self, nom, prenom, date_nais):
        self.nom = nom
        self.prenom = prenom
        self.date_nais = date_nais

    def Afficher(self):
        print(f"Nom: {self.nom}, Prénom: {self.prenom}, Né(e) en: {self.date_nais}", end=" ")

class Employe(Personne):
    def __init__(self, nom, prenom, date_nais, salaire):
        super().__init__(nom, prenom, date_nais)
        self.salaire = salaire

    def Afficher(self):
        super().Afficher()
        print(f", Salaire: {self.salaire}", end=" ")

class Chef(Employe):
    def __init__(self, nom, prenom, date_nais, salaire, service):
        super().__init__(nom, prenom, date_nais, salaire)
        self.service = service

    def Afficher(self):
        super().Afficher()
        print(f", Service: {self.service}", end=" ")

class Directeur(Chef):
    def __init__(self, nom, prenom, date_nais, salaire, service, societe):
        super().__init__(nom, prenom, date_nais, salaire, service)
        self.societe = societe

    def Afficher(self):
        super().Afficher()
        print(f", Société: {self.societe}", end=" ")

groupe = []

for i in range(5):
    groupe.append(Employe(f"Emp{i}", "Jean", 1990+i, 2000))

groupe.append(Chef("Chef1", "Paul", 1985, 3500, "RH"))
groupe.append(Chef("Chef2", "Marc", 1986, 3600, "IT"))
groupe.append(Directeur("Dir1", "Alice", 1975, 5000, "Direction", "MaBoite"))

print("\n--- Liste du personnel ---")
for p in groupe:
    p.Afficher()
    print() # Saut de ligne