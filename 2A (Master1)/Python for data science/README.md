Bienvenue sur le dépôt Github de notre projet Python réalisé dans le cadre de notre 2A à l'ENSAE, pour le cours de Python pour la Data Science !  

Ce projet intitulé **"Analyse de sentiment sur les 100 meilleurs films au box office"** met en oeuvre les nombreux enseignements reçu à l'ENSAE, du webscrapping à la modélisation en passsant par le NLP et le Machine Learning. 

Tout projet commençant par une problématique, nous avons souhaité répondre à la question suivante: Peut-on prédire la note d'un film à partir d'informations de base et de l'analyse de commentaires sur celui-ci ? 

Pour répondre à cette question, nous avons décomposé notre travail en trois parties: 
- La **première partie** consistait à récupérer les données nécessaires à notre étude, à savoir les commentaires des films, la note globale, le budget, le box office, la durée du film et l'année de sortie en salle. Ces informations ont été recueillies
sur les pages correspondantes à l'aide d'un script python de webscrapping sur le site imdb: https://www.imdb.com/list/ls098466969/. A l'aide de la librairie Selenium, nous avons scrappé les 100 meilleurs films du site et récupéré 200 commentaires par film ainsi que les informations mentionnées ci-dessus.
- La **deuxième partie** de notre travail a été l'étape de pre-processing des données récoltées. En effet, les données brutes trouvées sur Internet nécessites un nettoyage conséquent.
  + Pour les commentaires, nous les avons pre-processés avec les méthodes classiques de nettoyages de string comme les expressions régulières.
  + Pour ce qui est des variables numériques, peu de changement ont été opérés sauf sur la conversion des heures et sur les changements des DataType quand cela était nécessaire afin de pouvoir réaliser des opérations algébriques.
- La **troisième partie** consistait en l'analyse statistique de notre base de données et en la mise en place de modèles pour tenter de répondre à notre question.
  + L'analyse statistique de nos données se penchent sur la répartitions des données, leurs corrélations ainsi que leur potentiel d'explicabilité dans le cadre de notre enquête
  + Le modèle, quant à lui, nous permet d'esquisser une première réponse à notre problématique et de mettre en avant certaines limites des méthodes statistiques employées.

En conclusion, les données récoltées nous ont permis d'expliquer une partie de la note attribuée au film sur le site IMDB mais certaines limites sont à souligner. En effet, le nombre de commentaires récoltés peut être insufisant dans la mesure où nous
n'en avons que 200 par film (voire moins). Par ailleurs, les variables explicatives en présence ne permettent pas de quantifier comment a été reçu le film, le contexte socio-économique et géopolitique de la période de sortie (Covid, guerre). Par ailleurs, le style du film n'a pas été pris en compte. De plus, les algorithmes de NLP utilisés sont loin d'être parfait.

Pour reproduire cette étude, il convient de:
+ Télécharger les librairies mentionnées dans le notebook principal main.ipynb
+ Télécharger "chromedriver" de la même version que votre version Chrome, et le mettre à coté de ce fichier jupyter dans les fichiers. Le lien de téléchargement est le suivant: https://googlechromelabs.github.io/chrome-for-testing/
+ Prendre en compte que la partie de WebScrapping prend 40 minutes à tourner et celle de pre-processing 20 min 
+ La contrainte de temps peut être contournée. Les fichiers sont joints au notebook et sont au format json et csv
+ Si vous ne voulez pas exécuter un seul notebook mais décomposer le projet en trois parties, comme ce que nous avons fait lors de la réalisation de celui-ci, les trois notebooks sont aussi joints au projet dans le dossier Notebooks splitted. Attention, il peut y avoir quelques cellules en moins ou différentes par rapport au fichier principal en raison de certaines modifications tardives.

Merci de nous avoir lu et bonne découverte de l'analyse de sentiment appliquée aux commentaires de film ! 
