include "globals.mzn";
int:m;
int:n;

set of int:dimension=1..(n div m)+1;
set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions
int: total = sum(i in I_ext, j in I_ext)(D[i,j]);
int: total_max = total * m;
var 0..total_max : total_dist;
array[C] of var 0..total_max: distance;

array[C] of var I:Start;
array[C] of var I:Finish;

array[dimension,C] of var 0..n:ass; 


constraint alldifferent(Start);
constraint alldifferent(Finish);
constraint forall(i,j in C) ((Start[i]!=Finish[j]));
constraint forall(i in C) (ass[1,i] == Start[i]);
constraint forall(i in C) (ass[(n div m)+1,i] == Finish[i]);

constraint forall(i in C) 
           ( sum(j in 1..(n div m) where ass[j,i]!=ass[j+1,i] ) (S[ass[j,i]]) <= L[i]  );

function var int: dist_sum(array[dimension,C] of var 0..n:ass,array[I_ext, I_ext] of int: D,int :courier) =
        let 
        {       
              var int: distance= sum(i in 1..(n div m) )(D[ass[i,courier],ass[i+1,courier]]) 
              +  D[n+1,ass[1,courier]]  + D[ass[(n div m)+1,courier],n+1]
         } in distance;

constraint forall(i in C) (   dist_sum(ass,D,i) == distance[i]);                

constraint among(n,ass,I);
constraint alldifferent_except_0(ass);
constraint total_dist==sum(i in C)(distance[i]);


