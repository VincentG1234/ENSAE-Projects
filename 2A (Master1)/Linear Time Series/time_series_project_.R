#### LINEAR TIME SERIES PROJECT SCRIPT ### 
require(tseries)
require(zoo)
require(ggplot2)
library(forecast)

# I - DATA

datafile <- "valeurs_mensuelles.csv"
data <- read.csv(datafile, sep=";")

colnames(data)

# Withdraw the third column, "Codes"
data <- data[,c(1,2)]              
# Withdraw the 4 first lines which are useless 
data <- data[4:nrow(data),]

#Rename columns
colnames(data) <- c("dates", "indice")                    # Rename the column names

dates_char <- as.character(data$dates)
dates_char[1] #
tail(dates_char,1)

#Change format
data$indice <- as.numeric(data$indice)
dates <- as.yearmon(seq(from=1990+1/12, to=2024+2/12, by=1/12)) #
indice <- zoo(rev(data$indice), order.by=dates)

# Différentiation d'ordre 1
dindice <- diff(indice,1)

# On retire la première observation pour pouvoir former un dataframe avec dindice
indice <- indice[2:length(indice)]
dates <- dates[2:length(dates)]

# Création d'un dataframe (mieux pour utiliser ggplot2 proprement)
df <- data.frame(dates,indice,dindice)

# Plot of the time serie
ggplot(data=df, aes(x = dates, y = indice)) +
  geom_line(color = "blue") +
  labs(title = "Indice CVS-CJO of electricity production (base 100 in 2021)", x = "Date", y = "indice")+
  theme_gray()

#The series in level seems to be very persistent and to have a non linear trend, perhaps non deterministic. It looks
#a lot like a random walk. However, in first difference, it seems relatively stable around a null constant and could
#be stationary. The indice series is probably I(1).

# Plot of différentiated serie
ggplot(data=df, aes(x = dates, y = dindice)) +
  geom_line(color = "blue") +
  labs(title = "differentiated series", x = "Date", y = "indice")+
  theme_gray()

### -> La série différenciée semble stationnaire

# On retire les événements du COVID, comme suggéré par nos chargés de TD
indice <-indice[1:360,]
dindice <- dindice[1:360,]

desaison <- indice-lag(indice,-12)

# Plot of the time serie without COVID
ggplot(data=df[1:360,], aes(x = dates, y = indice)) +
  geom_line(color = "blue") +
  labs(title = "Indice electricity production without COVID", x = "Date", y = "indice")+
  theme_gray()


# Plot of différentiated serie without COVID
ggplot(data=df[1:360,], aes(x = dates, y = dindice)) +
  geom_line(color = "blue") +
  labs(title = "differentiated series without COVID", x = "Date", y = "indice")+
  theme_gray()

# Certes la base 100 est en 2021, et cette année n'est plus dans notre série,
# mais on peut voir que la série vaut à plusieurs reprises 100 entre 2000 et 2020
# On peut alors se dire que la nouvelle base est à un moment entre 2000 et 2020

#Before performing the unit root tests, we need to check if there is an intercept and / or a non null linear trend.
summary(lm(dindice~dates, data=df[1:360,]))
# there is no trend or intercept


### Test statistique pour la stationnarité ###

# test de Dickey-Fuller augmenté (ADF)
adf_test <- adf.test(dindice, alternative = "stationary")
adf_test
# p-value = 0.01. On rejette aux seuils usuels l'hypothèse nulle de non stationnarité.

# test KPSS
kpss_test <- kpss.test(dindice)
kpss_test
# P-value = 0.1. Nous échouons à rejeter l'hypothèse nulle que la série est 
# stationnaire aux seuils de 5% et 1%. Ambigu pour le seuil de 10%.

# Nous pouvons conclure que notre série est stationnaire. indice est I(1)


# II-ARMA

#A) Selection du modele 

par(mfrow=c(1,2))
acf(dindice,25);pacf(dindice,25)
#The autocorrélation of order 1 is around -0.3, which is small and not close
#to 1. The series seems stationary

# Par lecture graphique
pmax=11;qmax=10

#B) critère d'inforamtion

# On test l'ensemble des combinaisons possibles
pqs <- expand.grid(0:pmax,0:qmax) #combinaisons possibles de p<=p* et q<=q*
mat <- matrix(0, nrow=pmax+1, ncol=qmax+1)
rownames(mat) <- paste0("p=",0:pmax) #renomme les lignes
colnames(mat) <- paste0("q=",0:qmax) #renomme les colonnes
AICs <- mat #matrice ou assigner les AIC
BICs <- mat #matrice ou assigner les BIC

### TAKE TIME TO BE EXECUTED ###
for (row in 1:dim(pqs)[1]){
  p <- pqs[row,1]
  q <- pqs[row,2]
  estim <- try(arima(dindice,c(p,0,q), include.mean=F)) #tente d'estimer l'ARIMA
  AICs[p+1,q+1] <- if (class(estim)=="try-error") NA else estim$aic
  BICs[p+1,q+1] <- if (class(estim)=="try-error") NA else BIC(estim)
}
AICs
BICs
AICs==min(AICs)
# the AIC recommends p=8 et q=10
BICs==min(BICs)
## the BIC recommends q=p=1 

arima111 <- arima(indice, c(1,1,1), include.mean=F)
arima8110 <- arima(indice, c(8,1,10), include.mean=F)


# C) Significativité des modeles
arima111

# The ARIMA(1,1,1) coefficients are statistically significatif (the ratio between the estimated coefficiant and the standard
# error is higher in absolute value than 1.96), the ARIMA(1,1,1) is well adjusted.

arima8110
# The ARIMA(8,1,10) coefficients are not all statistically significatif. The model is not well ajusted


# D) Autocorrélation des résidus

plot(arima111$residuals)
plot(arima8110$residuals)

# Fonction pour effectuer le test de non autocorrélation des résidus pour différents lags
Qtests <- function(series, k, fitdf=0) {
  pvals <- apply(matrix(1:k), 1, FUN=function(l) {
    pval <- if (l<=fitdf) NA else Box.test(series, lag=l, type="Ljung-Box", fitdf=fitdf)$p.value
    return(c("lag"=l,"pval"=pval))
  })
  return(t(pvals))
}

Qtests(arima111$residuals, 25, 1) #LB tests for orders 1 to 24
# Nous rejettons l'hypothèse Ho aux lag 12-13 seulement au seuil de 5%.
# Le modele ARIMA(1,1,1) n'est pas valide. Mais il reste assez bien. Nous le garderons si nous trouvons rien de mieux

Qtests(arima8110$residuals, 25, 18)
# Le modele ARIMA(8;1,10) n'est pas valide

# Testons les modèles tels que q<= q* et p <=p* et tels que les modèles soient valides:
# Pour ce faire, on reprend les fonctions du TD4

#test function of individual statistical significance of the coefficients
signif <- function(estim){
  coef <- estim$coef
  se <- sqrt(diag(estim$var.coef))
  t <- coef/se
  pval <- (1-pnorm(abs(t)))*2
  return(rbind(coef,se,pval))
}

##function to print the tests for the ARIMA model selection
arimafit <- function(estim){
  
  adjust <- round(signif(estim),3)
  pvals <- Qtests(estim$residuals,24,length(estim$coef)-1)
  pvals <- matrix(apply(matrix(1:24,nrow=6),2,function(c) round(pvals[c,],3)),nrow=6)
  colnames(pvals) <- rep(c("lag", "pval"),4)
  cat("coefficients nullity tests :\n")
  print(adjust)
  cat("\n tests of autocorrelation of the residuals : \n")
  print(pvals)
}

# Essayons d'abord des modèles parcimonieux, avec p<=3 et q<= 6 par exemple

# rejette tous les modèles avec q=0

# rejette tous les modèles avec q=1 (ARIMA(1,1,1) reste le meileur)

# rejette tous les modèles avec q=2

# ARIMA(3,1,3) est valide au seuil de 10% (p-values>=0.1)  (au moins 1) et coeff significatif sauf 1 mais son impact est faible puisque sa valeur est faible

# rejette tous les modèes avec q=4

# ARIMA(2,1,5) est pas mal (coeff significatif et validité à p-values>=0.08)

# rejette tous les modèles avec q=6

# En essayant pour des valeurs plus élévé, j'arrive à obtenir des modèles valides
#mais les coefficients ne sont plus vraiment significatif. Et il vaut mieux privilégier un modèle parcimonieux


estim <- arima(indice,c(1,1,1)); arimafit(estim)
estim <- arima(indice,c(3,1,3)); arimafit(estim)
estim <- arima(indice,c(2,1,5)); arimafit(estim)

# Hésitation entre ARIMA(3,1,3) et ARIMA(2,1,5)
# regardons les critères d'informations (voir plus haut):
#ARIMA(3,1,3) BIC:1759.943      AIC:1732.740
#ARIMA(2,1,5) BIC:1765.453     AIC:1734.364

# => ARIMA(3,1,3) semble préférable. Nous allons le garder.


arima313 <- arima(indice,c(3,1,3))

# E) Normalité des résidus 

residuals <- arima313$residuals
par(mfrow=c(1,1))
ggplot(data=df[1:360,], aes(x = dates, y = arima313$residuals)) +
  geom_line(color = "blue") +
  labs(title = "residuals of ARIMA(3,1,3)", x = "Date", y = "indice")+
  theme_gray()

hist(residuals, main="distribution of residuals")

# test de Shapiro HO: residu suit une normale
shapiro.test(residuals)
# On rejette HO (p-value<0.01)

qqnorm(residuals, datax=TRUE, main="QQ-plot residuals vs normal")
qqline(residuals,datax=TRUE)
# C'est surtout les queues de la distribution qui posent problème ("fat tail")

arima313
# On a: Yt = -0.84Yt-1 - 0.43Yt-2 + 0.3851 Yt-3 + e_t + e_t-1*0.37 - e_t-2*0.06 - e_t-3*0.78 
# Yt = dXt = Xt - Xt-1

# III - Prediction

T = length(dindice)

# Prévoir les valeurs futures
forecast_values <- forecast(arima313, h = 12)  # prévoir 12 périodes à l'avance, par exemple

# Afficher les prévisions
print(forecast_values)

# Tracer les prévisions
# Tracer uniquement les dernières valeurs de la série temporelle 
n <- 20
# Tracer uniquement les dernières valeurs de la série temporelle et les prévisions
plot(forecast_values, xlim = c(tail(time(indice), n)[1], tail(time(forecast_values$mean), 1)))

# La zone en gris clair est la zone de confiance à 95%
  
