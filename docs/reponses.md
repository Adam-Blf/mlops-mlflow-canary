# Réponses écrites

Adam Beloucif - M2 Mastère Data Engineering & IA - Model versioning and canary deployment

---

## 3. MLFlow features

*En faisant des recherches, lister les principales fonctionnalités de MLFlow. Notamment lister ce que permet de faire le tracking d'experimentations*

MLflow sert à gérer la vie d'un modèle, pas seulement à noter des résultats quelque part.

Le tracking garde chaque entraînement. Les hyperparamètres, les métriques, les fichiers produits, et il récupère tout seul le commit git et les versions des librairies. On peut logger une métrique à chaque époque avec un step, donc on récupère la courbe complète et pas juste le chiffre de fin. Il sait aussi rattacher le jeu de données utilisé au run.

Le reste de l'outil : le Model Registry qui catalogue les versions du modèle avec des alias, le format MLflow Model qui permet de recharger n'importe quel modèle avec le même appel, le déploiement direct en API avec mlflow models serve, et MLflow Projects pour figer l'environnement d'un entraînement. Les versions récentes ont ajouté beaucoup de choses autour des LLM, du traçage d'agents et un registre de prompts, mais je ne m'en suis pas servi ici.

L'intérêt du tracking, je l'ai vu concrètement sur le TP. Avec trois runs lancés je compare les rmse dans l'interface, et surtout je sais quel run a produit la version qui tourne dans mon service. Au bout de cinq entraînements sans ça, on ne sait plus lequel est lequel.

---

## 5. Canary deployment

*By making a search, explain what is the principle of canary deployment. Look also at what is blue/green deployment, are there similarities ? what are the differences ?*

Le canary consiste à envoyer la nouvelle version à une petite partie du trafic seulement. On commence vers 1 ou 5 pourcent, on regarde les erreurs, la latence, et pour un modèle la qualité des prédictions sur cette part. Si rien ne bouge on augmente, jusqu'à 100. Sinon on ramène tout sur l'ancienne version et seuls quelques utilisateurs auront vu le problème.

Pour que ça marche il faut faire tourner les deux versions en même temps, avoir quelque chose devant qui répartit le trafic, et du monitoring assez fin pour comparer les deux groupes. Sans cette dernière partie, un canary c'est juste un déploiement plus lent.

Le blue/green marche autrement. On a deux environnements complets, blue qui sert la prod et green où on déploie la nouvelle version. On valide green pendant que blue continue à tout encaisser, puis on bascule le trafic d'un coup. Blue reste allumé un moment, donc si ça casse le retour arrière est immédiat.

Les points communs sont assez évidents : dans les deux cas l'ancienne version reste disponible pour revenir en arrière, et il faut une couche de routage devant.

Ce qui change vraiment, c'est la granularité. Le canary y va progressivement, le blue/green bascule d'un bloc. Le blue/green demande deux environnements identiques pendant la transition, donc ça double la facture, mais son retour arrière est instantané. En contrepartie, une fois la bascule faite tout le monde est sur la nouvelle version, alors que le canary limite les dégâts à la part de trafic concernée. Et c'est le seul des deux qui permet de comparer les métriques des deux versions sur du vrai trafic avant de tout basculer.

Le truc à ne pas oublier : un canary vert ne veut pas dire que la version est bonne. Un bug qui n'apparaît qu'à pleine charge, une fuite mémoire ou un pool de connexions saturé, ne se verra jamais sur 5 pourcent du trafic. Et ce qu'on ne mesure pas ne déclenchera aucune alerte.

---

## 8. Securing MLFlow

*Suppose you put your MLFlow server on a public available VM, does MLFlow offer methods to restrict access ? If not what could you do ?*

Oui, il y a quelque chose, mais c'est minimal et ce n'est pas activé par défaut.

MLflow embarque une authentification HTTP basique, qu'on active avec `mlflow server --app-name basic-auth`. Ça ajoute des utilisateurs et des permissions (READ, EDIT, MANAGE) sur les expériences et les modèles enregistrés, avec un compte admin créé au premier lancement. Les comptes vivent dans un `basic_auth.db` posé à côté du serveur, et la doc dit elle-même que les identifiants y sont stockés en clair, sans politique de mot de passe ni verrouillage de compte.

Par défaut, il n'y a rien du tout. Un MLflow posé sur une VM publique, c'est lecture et écriture ouvertes à n'importe qui : lister les expériences, télécharger les artefacts donc les modèles eux-mêmes, pousser une version, supprimer le registre. Et si le serveur tourne avec `--serve-artifacts`, le stockage d'artefacts passe par la même porte.

Ce que je ferais :

Ne pas exposer le port 5000. Le serveur écoute en local et un reverse proxy, nginx ou Traefik, fait le TLS et l'authentification devant. MLflow ne fait pas de HTTPS tout seul, donc sans ça le basic auth balade le mot de passe en clair sur le réseau.

Bloquer au niveau réseau, avec un security group ou un pare-feu limité aux IP de l'équipe. Le mieux restant de ne rien exposer du tout et de passer par un VPN ou un tunnel SSH.

Activer basic-auth quand même, pour avoir la séparation des droits à l'intérieur de MLflow, mais en la traitant comme une deuxième couche et pas comme la protection principale.

Mettre les artefacts sur du S3 ou du MinIO avec ses propres droits, plutôt que sur le disque du serveur.

Et ne jamais logger de secret en paramètre ou en tag : tout ce qui est tracé est lisible par qui a un accès en lecture.

Dernier point, MLflow a eu des vulnérabilités publiées sur la gestion des chemins d'artefacts, du type lecture de fichier arbitraire. Un serveur exposé doit donc aussi être tenu à jour, pas seulement mis derrière un mot de passe.
