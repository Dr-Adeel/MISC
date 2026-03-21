from abc import ABC, abstractmethod
from typing import List

# Étape 2 : Composition (L'Historique)
class Transaction:
    def __init__(self, description: str, montant: float):
        self.description = description
        self.montant = montant

    def __str__(self):
        return f"Transaction: {self.description} | Montant: {self.montant}"

# Étape 1 : Abstraction et Encapsulation
class Compte(ABC):
    def __init__(self, titulaire: str, solde_initial: float = 0):
        self.titulaire = titulaire
        self.__solde = solde_initial  # Attribut privé

    def consulter_solde(self):
        print(f"Solde actuel de {self.titulaire}: {self.__solde}")
        return self.__solde

    # Méthode protégée pour modifier le solde au sein des classes filles
    def _modifier_solde(self, montant: float):
        self.__solde += montant

    # Étape 4 : Surcharge (Overloading via arguments par défaut)
    def deposer(self, montant: float, note: str = "Dépôt"):
        if montant > 0:
            self._modifier_solde(montant)
            print(f"Déposé: {montant} ({note})")
        else:
            print("Le montant du dépôt doit être positif.")

    @abstractmethod
    def retirer(self, montant: float):
        pass

# Étape 3 : Héritage et Spécialisation
class CompteCourant(Compte):
    def __init__(self, titulaire: str, solde_initial: float = 0):
        super().__init__(titulaire, solde_initial)
        self.historique: List[Transaction] = [] # Composition

    def retirer(self, montant: float):
        # On autorise le découvert simple pour l'exemple, ou on vérifie le solde
        current_solde = self.consulter_solde()
        if current_solde >= montant:
            self._modifier_solde(-montant)
            self.historique.append(Transaction(f"Retrait", -montant))
            print(f"Retiré: {montant}")
        else:
            print("Fonds insuffisants.")

    # Surcharge de deposer pour enregistrer dans l'historique
    def deposer(self, montant: float, note: str = "Dépôt"):
        super().deposer(montant, note)
        self.historique.append(Transaction(f"{note}", montant))

    def virement(self, destinataire: 'Compte', montant: float):
        if self.consulter_solde() >= montant:
            self.retirer(montant) # Enregistre déjà le retrait dans l'historique
            destinataire.deposer(montant, f"Virement de {self.titulaire}")
            # On pourrait modifier la dernière transaction pour préciser que c'est un virement
            print(f"Virement de {montant} effectué vers {destinataire.titulaire}")
        else:
            print("Solde insuffisant pour le virement.")

    def afficher_historique(self):
        print(f"--- Historique du compte de {self.titulaire} ---")
        for t in self.historique:
            print(t)
        print("---------------------------------------------")


class CompteEpargne(Compte):
    def __init__(self, titulaire: str, solde_initial: float = 0, taux_interet: float = 2.0):
        super().__init__(titulaire, solde_initial)
        self.taux_interet = taux_interet

    def retirer(self, montant: float):
        if self.consulter_solde() >= montant:
            self._modifier_solde(-montant)
            print(f"Retiré (Epargne): {montant}")
        else:
            print("Fonds insuffisants sur le compte épargne.")

    def calculer_interets(self):
        solde_actuel = self.consulter_solde()
        interets = solde_actuel * (self.taux_interet / 100)
        self._modifier_solde(interets)
        print(f"Intérêts calculés: +{interets}. Nouveau solde: {self.consulter_solde()}")

# La Classe Client : Orchestration et Association (Agrégation)
class Client:
    def __init__(self, nom: str):
        self.nom = nom
        self.comptes: List[Compte] = [] # Agrégation

    def ajouter_compte(self, compte: Compte):
        self.comptes.append(compte)

    def afficher_synthese(self):
        print(f"\nSynthèse pour le client {self.nom} :")
        for compte in self.comptes:
            print(f"- Type: {type(compte).__name__}, Solde: {compte.consulter_solde()}")

# --- Code de test ---
if __name__ == "__main__":
    print("=== Création du système bancaire ===\n")

    # Création du client
    client1 = Client("Alice")

    # Création des comptes
    cc = CompteCourant("Alice", 1000)
    ce = CompteEpargne("Alice", 5000, 3.0)

    # Association
    client1.ajouter_compte(cc)
    client1.ajouter_compte(ce)

    # Opérations sur Compte Courant
    print("\n--- Opérations Compte Courant ---")
    cc.deposer(500, "Salaire")
    cc.retirer(200)
    cc.afficher_historique()

    # Opérations sur Compte Epargne
    print("\n--- Opérations Compte Epargne ---")
    ce.calculer_interets()

    # Virement
    print("\n--- Virement CC vers un autre compte ---")
    compte_tiers = CompteCourant("Bob", 0)
    cc.virement(compte_tiers, 300)

    # Synthèse finale
    client1.afficher_synthese()
    print("\nSolde Bob:", compte_tiers.consulter_solde())
