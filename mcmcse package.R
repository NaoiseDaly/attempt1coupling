rm(list = ls())
dev.off()
require("mcmcse")
# setwd(
#   r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\keep_safe\3D-MVN-sample)"
# )
# setwd(
#     r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\logs_and_data)"
#    )

# B = 940#518 #read it from log file
# # l.chain <- read.csv("high_autocorrelated_mvn-P3-Seed42_lagged_chain.csv"
# #                     , header = F)
# # nl.chain<- read.csv("high_autocorrelated_mvn-P3-Seed42_unlagged_chain.csv"
# #                     , header = F)
# l.chain <- read.csv("high_autocorrelated_mvn-P3-Seed42_lagged_chain_2025-03-05 Wed 13-01.csv"
#                     , header = F)
# nl.chain<- read.csv("high_autocorrelated_mvn-P3-Seed42_unlagged_chain2025-03-05 Wed 13-01.csv"
#                     , header = F)
# l.chain <- l.chain[B:dim(l.chain)[1], ]
# nl.chain <- nl.chain[B:dim(nl.chain)[1], ]

# dim(l.chain)


# chain <- l.chain
# #brief look
# summary(l.chain)
# for (c in 1:ncol(l.chain)){
#   plot(
#     l.chain[,c], col = "blue", type = "l", lwd = 1.5
#    ,xlab = "time", ylab = paste("component ",c)
#        )
# }

# nrow(l.chain);multiESS(l.chain);ess(l.chain)
# nrow(nl.chain);multiESS(nl.chain);ess(nl.chain)


rm(list = ls())
dev.off()
old.par <- par(no.readonly = T);back <- function(){ par(old.par)}
require("mcmcse")
setwd(
    r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\keep_safe\good samples\8schools example)"
   )

long = read.csv("8schools_long_chain_2025-03-16 Sun 20-14.csv", header = F)
short = read.csv("8schools_short_chain_2025-03-16 Sun 20-14.csv", header = F)
dim(long);dim(short)
# long <- read.csv("high_autocorrelated_mvn-P3-Seed42_lagged_chain_2025-03-05 Wed 13-01.csv"
#                     , header = F)
# short<- read.csv("high_autocorrelated_mvn-P3-Seed42_unlagged_chain2025-03-05 Wed 13-01.csv"
#                     , header = F)

# #remove 10% burn in 
# short = short[dim(short)[1]%/%10:dim(short)[1], ]
# long = long[dim(long)[1]%/%10:dim(long)[1], ]


lst = list(long = long, short= short)
for (i in lst){
  par(mfrow = c(min(ncol(chain),4),1))
  for (c in colnames(chain)){
    plot(
      chain[,c], col = "blue", type = "l", lwd = 1.5
      ,xlab = "time", ylab = paste("component ",c)
    )
  }
  }
chain <- long
par(mfrow = c(min(ncol(chain),4),1))
for (c in colnames(chain)){
  plot(
        chain[,c], col = "blue", type = "l", lwd = 1.5
       ,xlab = "time", ylab = paste("component ",c)
           )
}

nrow(long);multiESS(long);min(ess(long))
nrow(short);multiESS(short);min(ess(short))

steps = nrow(long)- ncol(long)
mv.ess <- numeric(steps)
causedWarning = numeric(steps)
for (t in 1:steps) {
  output <- tryCatch( multiESS(long[t:nrow(long),])
                      , warning = function(w) c(multiESS(long[t:nrow(long),]), 1) )
  if (length(output) >1){
    causedWarning[t] = 1
    mv.ess[t] = output[1]
  }
  else{
    mv.ess[t] = output
  }
  
}
back()
plot(mv.ess, col = c("black","red")[as.factor(causedWarning)], pch = 20)
plot(which(causedWarning == 0), mv.ess[causedWarning == 0]
     ,main = "multi ESS on windows starting from t"
     ,xlab = "t", ylab = "MV ESS"
     ,pch = 20)
#notice for Sun 20-14 datset 
multiESS(long[45:nrow(long),])# fails



uni_ESS_long = matrix(rep(0, steps*2), ncol = 2)
colnames(uni_ESS_long) <- c("uni_ESS", "warning")
func <- function(t){ min(ess(long[t:nrow(long),])) }
for (t in 1:steps) {
  output <- tryCatch( func(t)
                      , warning = function(w) c(func(t), T) )
  if (length(output) >1){

    uni_ESS_long[t, ] = c(output[1],T)
  }
  else{
    uni_ESS_long[t, 1] = output
  }
  
}
cols = c("black","red")[as.factor(uni_ESS_long[,"warning"])]
plot(uni_ESS_long[,"uni_ESS"], type = "l")


