rm(list = ls())
dev.off()
require("mcmcse")
# setwd(
#   r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\keep_safe\3D-MVN-sample)"
# )
setwd(
    r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\logs_and_data)"
   )

B = 940#518 #read it from log file
# l.chain <- read.csv("high_autocorrelated_mvn-P3-Seed42_lagged_chain.csv"
#                     , header = F)
# nl.chain<- read.csv("high_autocorrelated_mvn-P3-Seed42_unlagged_chain.csv"
#                     , header = F)
l.chain <- read.csv("high_autocorrelated_mvn-P3-Seed42_lagged_chain_2025-03-05 Wed 13-01.csv"
                    , header = F)
nl.chain<- read.csv("high_autocorrelated_mvn-P3-Seed42_unlagged_chain2025-03-05 Wed 13-01.csv"
                    , header = F)
l.chain <- l.chain[B:dim(l.chain)[1], ]
nl.chain <- nl.chain[B:dim(nl.chain)[1], ]

dim(l.chain)


chain <- l.chain
#brief look
summary(l.chain)
for (c in 1:ncol(l.chain)){
  plot(
    l.chain[,c], col = "blue", type = "l", lwd = 1.5
   ,xlab = "time", ylab = paste("component ",c)
       )
}

nrow(l.chain);multiESS(l.chain);ess(l.chain)
nrow(nl.chain);multiESS(nl.chain);ess(nl.chain)




