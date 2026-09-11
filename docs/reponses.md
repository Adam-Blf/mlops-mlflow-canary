# Réponses écrites

Adam Beloucif - M2 Mastère Data Engineering & IA - Model versioning and canary deployment

Références : documentation officielle MLflow (mlflow.org/docs/latest, ligne MLflow 3, version 3.3.0 au moment de la rédaction).

---

## Les principales fonctionnalités de MLflow, et ce que permet le tracking d'expérimentations

MLflow est souvent présenté comme quatre briques : Tracking, Projects, Models et Registry. Cette description date. La documentation actuelle organise le produit en deux domaines distincts : le machine learning classique d'un côté (suivi d'expériences, empaquetage, registre, déploiement), et les applications génératives de l'autre (traçage d'agents et de LLM, gestion des prompts, évaluation par juge LLM, passerelle d'accès aux modèles). MLflow 3 a d'ailleurs retiré certains composants historiques, notamment Recipes et le Deployments Server.

Les fonctionnalités principales, telles qu'elles existent aujourd'hui :

- **Tracking**, le suivi d'expérimentations, détaillé plus bas.
- **Empaquetage des modèles et flavors.** MLflow définit un format standard, le « MLflow Model », qui encapsule un modèle entraîné avec sa signature d'entrée et de sortie, ses dépendances exactes et un exemple d'entrée. Ce format est reconnu par de nombreux frameworks (scikit-learn, PyTorch, XGBoost, transformers, LangChain), ce qui permet de recharger un modèle sans savoir avec quoi il a été produit.
- **Model Registry.** Un catalogue centralisé des versions de modèles, avec alias, tags et description. Les alias remplacent depuis la version 2.9 les anciens « stages » du type Staging et Production, aujourd'hui dépréciés.
- **Déploiement.** `mlflow models serve` expose un modèle en API REST locale, `mlflow models build-docker` génère une image conteneur, et des plugins ciblent Databricks, SageMaker ou Kubernetes.
- **Évaluation** avec `mlflow.evaluate`, sur des métriques classiques de classification et de régression, et côté génératif sur des critères de qualité de réponse évalués par un LLM juge (hallucination, pertinence, toxicité).
- **Traçage des applications génératives.** MLflow capture l'exécution complète d'un agent ou d'un pipeline RAG : prompts, appels d'outils, étapes de récupération, latence par étape. L'implémentation s'appuie sur OpenTelemetry, ce qui la rend indépendante du fournisseur de modèle.
- **Registre de prompts.** Versioning et comparaison de templates de prompts, traités comme des artefacts à part entière.
- **AI Gateway.** Une passerelle unique devant plusieurs fournisseurs de modèles, pour centraliser les clés, suivre les coûts et appliquer des contrôles d'accès.
- **Projects.** Un format d'empaquetage du code d'entraînement lui-même, via un fichier `MLproject`, qui fige l'environnement et les points d'entrée pour rejouer un entraînement à l'identique ailleurs.

### Ce que le tracking enregistre, run par run

- Les **paramètres** (`log_param`, `log_params`) : hyperparamètres et configuration, par exemple le taux d'apprentissage, la taille de batch ou la profondeur d'un arbre.
- Les **métriques** (`log_metric`, `log_metrics`), avec un argument `step` qui permet d'enregistrer leur évolution au fil des époques. On obtient donc de vraies courbes d'apprentissage, pas seulement une valeur finale.
- Les **artefacts** (`log_artifact`) : tout fichier produit par le run, graphiques, matrice de confusion, checkpoints, et le modèle sérialisé lui-même.
- Les **tags**, métadonnées libres servant à filtrer et à rechercher, par exemple le propriétaire du run ou la version du jeu de données.
- Le **code source et la version git**. MLflow enregistre automatiquement le hash du commit courant et le fichier à l'origine du run, ce qui raccroche chaque résultat à un état précis du code.
- L'**environnement d'exécution** : les versions des bibliothèques utilisées, capturées au moment du run.
- Le **jeu de données d'entrée**, via `log_input`, qui relie un `DatasetInput` au run avec sa source, son schéma et son empreinte.

### Ce que cela permet concrètement

Comparer des runs entre eux dans l'interface ou par l'API, avec tri, filtres et graphes en coordonnées parallèles, pour isoler l'effet d'un hyperparamètre sur une métrique. Reproduire un run à l'identique, puisque le code, l'environnement et les données sont tous référencés au même endroit. Retrouver automatiquement le meilleur modèle d'une expérience avec `mlflow.search_runs`, puis l'enregistrer directement au registre depuis ce run. Et surtout, auditer un modèle en production en remontant jusqu'à l'expérience qui l'a produit, ce qui reste la seule façon d'expliquer après coup pourquoi un modèle se comporte comme il se comporte.

C'est ce dernier point qui justifie l'outil. Un tableur de résultats fait illusion tant qu'on est seul sur un projet ; il ne survit pas à la première question du type « avec quelles données exactement ce modèle a-t-il été entraîné ».

---

## Canary deployment and blue/green deployment

*La question étant posée en anglais, la réponse suit dans la même langue.*

### Canary deployment

The principle is to expose a new version to a small, controlled share of production traffic before exposing it to everyone. A typical rollout goes one percent, then five, then ten, then fifty, while the remaining traffic keeps hitting the current stable version. At each step the team watches error rates, latency, business metrics, and in a machine learning context the quality metrics computed on that live slice. If the new version behaves, traffic is increased until it serves everything, which is the promotion. If a regression shows up, traffic is routed back to the old version, and only a small share of users was ever exposed.

This needs three things that are easy to underestimate : infrastructure able to run both versions simultaneously, a routing layer able to split traffic dynamically (load balancer, service mesh, or feature flags), and monitoring granular enough to compare the two populations while both are live. Without the third one, a canary is just a slower deployment.

### Blue/green deployment

Two complete environments run side by side. Blue serves production, green receives the new version and is validated while blue still handles every request. Once green is considered ready, traffic is switched over in a single cut, usually at the load balancer or DNS level. Blue is kept warm for a while as an immediate fallback : if something breaks after the switch, traffic goes straight back to it.

### Similarities

Both reduce the risk of a bad release reaching every user at once, both keep the previous version available for a fast rollback, and both require a routing layer in front of the versions rather than a single fixed target.

### Differences

- **Traffic granularity.** Canary shifts traffic progressively and in fine increments. Blue/green switches everything at once, in a discrete cut over.
- **Infrastructure cost.** Canary only needs enough extra capacity for the share of traffic it serves. Blue/green requires two full environments of identical capacity during the transition, which roughly doubles the bill.
- **Rollback speed.** Blue/green rollback is essentially instantaneous, since the untouched blue environment is still running. Canary rollback is fast too, but it means resetting a traffic split, and if the rollout had already reached a high percentage, more users were affected before anyone decided to stop.
- **Blast radius.** Canary deliberately contains the blast radius to the slice currently routed to the new version. Blue/green does not contain it at all once the switch has happened : everyone moves together, and the safety net is the speed of the rollback, not the size of the exposure.
- **Data and schema compatibility.** Both strategies run two versions against the same data layer at some point, so both require forward and backward compatible schemas. The constraint simply bites harder with canary, because the two versions coexist for hours or days rather than minutes, and data written by one version must stay readable by the other for that whole period.
- **Statistical detection before full rollout.** This is the real differentiator. Canary lets you compare metrics between the canary population and the baseline population while both receive live traffic, so a regression can be detected with actual evidence before full exposure. Blue/green validates green before the switch, in a pre production or smoke test phase, which never tells you how the new version behaves on real users.

### What canary deployment does not protect against

A canary only catches what is visible on a small, low traffic slice. It will not catch a bug that only appears at full production load : a memory leak that takes hours to matter, connection pool exhaustion, or a race condition that needs real concurrency to trigger. The canary slice never reaches that scale, so those defects pass the gate and surface right after promotion.

It will not catch a slow drift either, nor anything the monitoring does not measure. A model whose accuracy degrades on a subpopulation that no metric covers, or a fairness issue that only becomes visible over a longer window than the canary phase lasts, goes through unnoticed. A canary phase that looks perfectly healthy proves that nothing measured went wrong, which is a narrower statement than it appears.

---

## Les autres fonctions de MLflow, hors versioning d'expérimentations

Le suivi d'expérimentations est la porte d'entrée de MLflow, mais ce n'est pas ce qui en fait une plateforme. Les fonctions suivantes couvrent le reste du cycle de vie.

**La gestion du cycle de vie dans le Model Registry.** Le registre ne se contente pas d'empiler des versions. Il permet de les nommer par alias, par exemple `champion` et `challenger`, d'y attacher des tags et une description, et de garder l'historique de qui a promu quelle version et quand. C'est ce qui rend un déploiement traçable : le service ne charge pas « le modèle », il charge une version identifiée du registre. Les alias ont remplacé les anciens stages Staging et Production, dépréciés depuis la version 2.9, précisément parce qu'un vocabulaire figé à trois niveaux ne survit pas à une organisation réelle.

**L'empaquetage portable.** Le format MLflow Model embarque le modèle, sa signature, ses dépendances et un exemple d'entrée. Un modèle scikit-learn, un modèle PyTorch et une chaîne LangChain se rechargent tous avec le même appel `mlflow.pyfunc.load_model`. Pour un service web comme celui de ce projet, c'est exactement ce qui permet d'écrire le chargement une seule fois et de changer de framework d'entraînement sans toucher au service.

**Le déploiement effectif.** `mlflow models serve` expose un modèle en API REST en une commande, ce qui suffit à valider un modèle avant d'écrire quoi que ce soit. `mlflow models build-docker` produit une image conteneur. Des plugins couvrent Databricks Model Serving, SageMaker et Kubernetes. Le Deployments Server historique a été retiré dans MLflow 3.

**MLflow Projects.** Un fichier `MLproject` fige les points d'entrée et l'environnement d'un entraînement, pour le rejouer à l'identique sur une autre machine. C'est le pendant, côté code, de ce que le registre fait côté modèle.

**L'évaluation.** `mlflow.evaluate` calcule automatiquement les métriques standard d'un modèle de classification ou de régression sur un jeu d'évaluation, et produit les artefacts associés. Côté applications génératives, la même fonction évalue des réponses de LLM avec des juges préconstruits qui détectent l'hallucination, le manque de pertinence ou la toxicité, ou avec des métriques définies sur mesure.

**Le traçage des applications génératives.** MLflow enregistre l'exécution complète d'un agent ou d'un pipeline de récupération augmentée : prompts envoyés, appels d'outils, documents récupérés, latence de chaque étape. L'implémentation suit le standard OpenTelemetry, donc les traces sont exploitables dans un outil d'observabilité existant plutôt que dans un silo.

**Le registre de prompts.** Versioning, comparaison et recherche de templates de prompts depuis l'interface, avec des outils d'amélioration guidée par les données d'évaluation. Traiter un prompt comme un artefact versionné plutôt que comme une chaîne de caractères perdue dans le code relève du même raisonnement que celui qui a mené à versionner les modèles.

**L'AI Gateway.** Une passerelle unique devant plusieurs fournisseurs de modèles, qui centralise les clés d'API, suit les coûts et applique des contrôles d'accès, au lieu de laisser chaque application appeler chaque fournisseur directement.

En résumé, MLflow a dépassé le statut d'outil de suivi d'expériences. Il couvre l'empaquetage, le catalogue et la promotion des versions, le déploiement, l'évaluation de la qualité, et, pour les applications à base de LLM, le traçage d'exécution, la gestion des prompts et la gouvernance des accès.
