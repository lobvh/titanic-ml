library(parallel)

#detects number of cores available to use for parallel package
nCores <- detectCores(logical = FALSE)
cat(nCores, " cores detected.")  

# detect threads with parallel()
nThreads<- detectCores(logical = TRUE)
cat(nThreads, " threads detected.")

# Create doSNOW compute cluster (try 64)
# One can increase up to 128 nodes
# Each node requires 44 Mbyte RAM under WINDOWS.
cluster <- makeCluster(128, type = "SOCK")
class(cluster);
