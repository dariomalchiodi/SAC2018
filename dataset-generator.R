####################################################
# Generate a dataset from RDFMiner's results for SVM learning
# (c) 2016 Andrea G. B. Tettamanzi
####################################################

library(SPARQL) # SPARQL querying package

# Read the table with the statistics produced by RDF Miner w/o time capping
d <- read.table("../table.txt", header = TRUE)

# Compute the ARI for all axioms:
d$ari <- d$nec + d$poss - 1

fix.URI = function(uri)
{
  if(substr(uri, 1, 1) == "<")
    uri <- paste(uri, ">", sep = "")
  uri
}

# Extract the subclass and superclass from each axiom
s <- strsplit(as.character(d$axiom), "[ ()]")
for(i in 1:length(d$axiom))
{
  d$C[i] <- fix.URI(s[[i]][2]) 
  d$D[i] <- fix.URI(s[[i]][3])
}

# Define the DBpedia endpoint
endpoint <- "http://DBpedia.org/sparql"

# Define a list of common prefixes
prefix <-
"PREFIX owl: <http://www.w3.org/2002/07/owl#> \
 PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
 PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> \
 PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \
 PREFIX foaf: <http://xmlns.com/foaf/0.1/> \
 PREFIX dc: <http://purl.org/dc/elements/1.1/> \
 PREFIX : <http://dbpedia.org/resource/> \
 PREFIX dbpedia2: <http://dbpedia.org/property/> \
 PREFIX dbpedia: <http://dbpedia.org/> \
 PREFIX skos: <http://www.w3.org/2004/02/skos/core#> \
 PREFIX dbo: <http://dbpedia.org/ontology/>"

label <- character(2*length(d$axiom))
# The poss vector contains the possibility of each axiom
poss <- numeric(2*length(d$axiom))
for(i in 1:(2*length(d$axiom)))
{
  poss[i] <- ifelse(i %% 2, 1 - d$nec[(i + 1) %/% 2], d$poss[(i + 1) %/% 2])
  label[i] <- paste(
    ifelse(i %% 2, "-", ""),
    as.character(d$axiom[(i + 1) %/% 2]),
    sep = "")
}

# Pre-compute an upper triangular matrix containing the denominators
# for the similarity computations between axioms.
denom <- matrix(nrow = length(d$axiom), ncol = length(d$axiom))
if(file.exists("denom.RData"))
  load("denom.RData")
cat("Pre-computing the denominators...\n")
for(i in 1:(length(d$axiom) - 1))
{
	if(is.na(denom[i, length(d$axiom)]))
	{
		cat(as.character(d$axiom[i]))
		for(j in (i + 1):length(d$axiom))
			if(is.na(denom[i, j]))
			{
			  # Compute the denominator
			  query <- sprintf("SELECT (count(DISTINCT ?x) AS ?n) \
			  WHERE { { ?x a %s . } UNION { ?x a %s . } }",
			  d$C[i], d$C[j])

			  qd <- SPARQL(endpoint, paste(prefix, query))
			  denom[i, j] <- qd$results$n[1]
			  cat(".")
			}
		# Play a sound to tell the user R is saving the data:
		system("paplay /usr/share/sounds/ubuntu/notifications/Blip.ogg", wait = FALSE)
		save(denom, file = "denom.RData")
		cat("\n")
	}
}
# Apparently, there are 91 zero entries in denom!
# sum(ifelse(is.na(denom), FALSE, denom==0))


# Construct the similarity matrix
# - Positive axioms correspond to even rows and columns,
# - negative axioms to odd rows and columns
sim <- matrix(nrow = 2*length(d$axiom), ncol = 2*length(d$axiom))
if(file.exists("sim.RData"))
  load("sim.RData")
for(i in 1:(2*length(d$axiom)))
{
  # If the row has not been completed yet...
#  if(is.na(sim[i, 2*length(d$axiom)]))
#  {
    for(j in 1:(2*length(d$axiom)))
    {
      if(i < j && is.na(sim[i, j]))
      {
        cat("Computing the similarity between", label[i], "and", label[j], "\n")
        left <- (i + 1) %/% 2
        right <- (j + 1) %/% 2

        if(left < right && denom[left, right] > 0)
        {
          # Compute the numerator
          left.conj = paste("?x a", d$C[left], ".",
            ifelse(i %% 2,
              paste("FILTER NOT EXISTS { ?x a", d$D[left], ". }"),
              paste("?x a", d$D[left], ".")))
          right.conj = paste("?x a", d$C[right], ".",
            ifelse(j %% 2,
              paste("FILTER NOT EXISTS { ?x a", d$D[right], ". }"),
              paste("?x a", d$D[right], ".")))
          query <- sprintf("SELECT (count(DISTINCT ?x) AS ?n) \
            WHERE { { %s } UNION { %s } }",
            left.conj, right.conj)
          # cat("Query: ", query, "\n")

          qd <- SPARQL(endpoint, paste(prefix, query))
          num <- qd$results$n[1]
 
          sim[i, j] <- num/denom[left, right]
          if(is.na(sim[i, j]))
            stop("NA: ", num, "/", denom[left, right], "...\n")
        }
        else # similarity between \phi and \neg\phi OR 0/0
          sim[i, j] <- 0.0

        # Save partial state every 10 cells
        # to avoid loosing all the work in case of error...
        if(j %% 10 == 0)
        {
          # Play a sound to tell the user R is saving the data:
          system("paplay /usr/share/sounds/ubuntu/notifications/Blip.ogg", wait = FALSE)
          save(sim, file = "sim.RData")
        }
      }
      else if(i == j)
        sim[i, j] <- 1.0
      else if(i > j)
        sim[i, j] <- sim[j, i] # the matrix is symmetric
    }
#    # Display a contour plot of the similarity matrix, just to check everything is OK
#    filled.contour(sim)
#  }
}
# Play a sound to tell the user R is saving the data:
system("paplay /usr/share/sounds/ubuntu/notifications/Blip.ogg", wait = FALSE)
save(sim, file = "sim.RData")

# Write the matrix to a file
# N.B.: no need to transpose, since sim is a symmetric matrix!
dataset <- cbind(poss, sim)
colnames(dataset) <- c("possibility", label)
write.table(dataset, file = "dataset.csv", sep = ", ", row.names = FALSE)
cat("Done.\n")

