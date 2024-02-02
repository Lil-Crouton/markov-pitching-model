# Methodology
This code models the states of a pitch count (0-0, 1-0, 1-1...) as a markov process. The rates of strikes and balls thrown in each count (the state transition probabilities) are computed for the league average and for individual pitchers allowing us to see how pitchers compare to the league. For instance we can compute the delta between the strikes Spencer Strider throws in a 1-1 count vs the league average to see how much better he is at throwing strikes.

One can store all of this information in state transition matrices, however I also created an interactive application with a visual graph view of this data. Check it out at thedatadugout.com

Data was sourced from retrosheet.org
