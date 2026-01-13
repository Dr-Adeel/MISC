# Exemple : Système Bancaire


## La Classe Compte

### Étape 1 : Abstraction et Encapsulation 
(La classe Base)Créez une classe abstraite Compte.

Attributs : '__solde' (privé), titulaire.

Méthodes : consulter_solde(), deposer(montant), retirer(montant). 
La méthode retirer doit être abstraite pour forcer les sous-classes à définir leurs propres règles de retrait (ex: découvert autorisé ou non).


### Étape 2 : Composition (L'Historique)

Créez une classe Transaction. Le CompteCourant doit contenir une liste d'objets Transaction. C'est une composition car les transactions appartiennent exclusivement à ce compte.


### Étape 3 : Héritage et Spécialisation

CompteCourant : Ajoute la méthode virement(destinataire, montant). Il enregistre chaque opération dans son historique.

CompteEpargne : Ajoute un attribut taux_interet. Créez une méthode calculer_interets() qui applique la formule :$$Solde_{nouveau} = Solde_{actuel} + (Solde_{actuel} \times \frac{Taux}{100})$$


### Étape 4 : Surcharge (Overloading)
En Python, la surcharge se simule souvent avec des arguments par défaut. Modifiez deposer() pour qu'on puisse soit déposer un montant simple, soit un montant avec une "note" ou "libellé" optionnel.

## La Classe Client : Orchestration et Association

La relation entre Client et Compte est une agrégation : si on supprime le client, les objets comptes pourraient techniquement exister (ou être transférés), mais le client "possède" une liste de références vers ces comptes.