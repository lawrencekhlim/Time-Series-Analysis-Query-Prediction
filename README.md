Main Goal:
Learning about the queries with the highest and lowest frequencies allows us to ignore the less important queries without sacrificing the most important queries. 
 
Formalizing the problem:
Let Q = {q1,q2,...qm} be the universe set of queries for m queries.
Let Y (t) be an indicator binary vector {0,1}^{m} on day t.
This means the value Y_1 (t) (or Y sub 1) is 1 if q1 is queried on day t and 0 otherwise.
This can even be expanded the encompass to the number of times a query is queried. (For instance if the query was queried twice, then the number would instead be two.)


We consider this problem a multi-label classification problem because every day is "labeled" to either be or not be queried with that set queries. Do I understand this correctly?


I originally thought that the problem formulation was the following: 
We construct a matrix X (t) composed of the last K days (up to day t), and map it to Y(t+1). 

Y = X * Z where X is a matrix with K rows and m columns a (K by m matrix), vector Z is of dimension K, and predictor Y is of dimension m queries.



But on multiple readings, I think the formulation is actually the following:
We construct a vector X (t) composed of then last K days (up to day t) and map it to Y (t+1).

Y = Z * X where Y is a vector of dimension m queries, X is a vector of dimension m*K previous queries, and Z is a matrix with m rows and m*K columns (a m by m*K matrix).

Is this formulation correct?

Minimizing the loss function  (Y - ZX) ^2 over all known values of X and Y will give us a better and better approximation for Y.


Minimizing the loss function & other math:
This part is my speculation on solving the math:
We can construct a matrix X (m*K by N) and matrix Y (m by N) consisting of all the known N data points do minimize the loss function to find Z.

           Y            =            Z                 *            X
(m by N) matrix = (m by m*K) matrix *   (m*K by N) matrix


We can minimize the loss function by finding the pseudo-inverse of matrix X (denoted here as X+) and multiplying it by Y to get Z.

           Y            *          X+                  =            Z                  *            X                  *            X+
(m by N) matrix * (N by m*K) matrix   =  (m by m*K) matrix *   (m*K by N) matrix * (N by m*K) matrix

so
Z  = Y * X+



Using Singular Value Decomposition (SVD), we can determine query correlation (and find the pseudo-inverse).


http://www.cs.utexas.edu/users/inderjit/public_papers/multilabel_icml14.pdf
However, the paper referenced formulates the problem as 
Y = Z^(T) * X               (in English, Y equals Z transpose times X)
The Z transpose is different, but similar to my formulation of the problem.

and in Claim 1 minimizes the error for (Y - X*Z)^2 using SVD with
Z = V * (Sigma)^(-1) * M for M = U^(T) * Y (truncated to rank K)

Substituting for M,

Z = V * (Sigma) ^(-1) * U^(T) * Y

V * (Sigma)^(-1)* U^(T) is just using SVD to find the inverse of X (although U transpose is truncated), so

Z = X+ * Y 
I think for right now I'll use the formulation as stated in the paper for the basis of my calculations. The claim minimizes the error for (Y - X*Z)^2, but does it minimize the error for (Y - Z^(T) * X)^2?



Question: I understand that SVD allows us to compress the size of the matrix by choosing only the greatest eigenvalues (and this in turn makes matrix operations of large size faster). However, I am not sure why we need dimensionality reduction (turning a large matrix X to a smaller one) or how SVD allows us to do dimensionality reduction. Could you clarify this for me? 

Also, I am not sure what we mean by Z being "low rank and in the optimization problem this will enforced using a regularizer"? I assume it has to do with r and lambda in the paper?

Choosing only the last K days for matrix X allows us to find temporal correlation. Does this mean we are interested in "temporary" correlations for the most recent K days (such as a short-term fad) or attempting to find cyclical correlations (for instance on Sundays people are more likely to go shopping)? I can see how this construction might allow us to predict both. Also, what goes in our decision of K?
