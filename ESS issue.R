rm(list = ls());dev.off()
old.par <- par(no.readonly = T);back <- function(){ par(old.par);par(pch=20)}
require("mcmcse")
setwd(
  r"(C:\Users\Naoise Daly\OneDrive - University College Cork\FYP\attempt1coupling\keep_safe\good samples\8schools example)"
)

#10% burn in has already been removed from both samples
long = read.csv("8schools_long_chain_2025-03-16 Sun 20-14.csv", header = F)
short = read.csv("8schools_short_chain_2025-03-16 Sun 20-14.csv", header = F)
dim(long);dim(short)

get_uni_ess <- function(dat){
  min(ess(dat))
}
get_mv_ess <- function(dat){
  multiESS(dat)
}

#sliding window 

sliding_window <- function(chain){
  window_size <- floor(dim(chain)[1]*.2)
  output <- NA
  for (t in 1:(nrow(chain)-window_size) ){
    window <- chain[t:(t+window_size),]
    uni <- get_uni_ess(window)
    mv <- get_mv_ess(window)
    output <- rbind(output, c(uni, mv))
  }
  colnames(output) <- c("Uni_ESS", "MV_ESS")
  return(output)
}

#reset graphics
back()


short_stats <- sliding_window(short)
long_stats <- sliding_window(long) #may take a minute

#Be aware in the difference in scale on the Y axis
#plot univariate
par(mfrow=c(1,2))
plot(short_stats[,1], col = "blue"
     , main = "Univariate ESS ", ylab = "n eff", xlab = "t")
plot(long_stats[,1], col = "purple"
     , main = "Univariate ESS", ylab = "n eff", xlab = "t")

#plot multivariate
plot(short_stats[,2], col = "blue"
     , main = "Multivariate ESS", ylab = "n eff", xlab = "t")
plot(long_stats[,2], col = "purple"
     , main = "Multivariate ESS", ylab = "n eff", xlab = "t")


back()
