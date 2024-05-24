include "globals.mzn";
int:m;
int:n;

int:dim = (n div m)+1+2;
set of int:dimension=1..dim;
set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions
int: total_max = sum(j in I_ext, i in 1..2)(D[i,j]);
var 0..total_max : total_dist;

array[I] of int: low_bounds = [D[n+1,i] +D[i,n+1] | i in 1..n] ;

int:low = min(low_bounds);

int: high = sum(i in 1..((n div m)+1))(D[i,i+1]) + D[n-m+1,1] + D[n-m,n+1];
array[C] of var low..high: distance;

var int:UB;
var int:LB ;
constraint UB = high;
constraint LB = low;

array[dimension,C] of var I_ext:ass; 
% array[C] of var I:n_obj;

% constraint sum(i in n_obj)(i) <= n;

% constraint forall(i in C)(
%               n_obj[i] = sum(j in 2..(dim-1) where ass[j,i] != (n+1) )(1));
              

% constraint symmetry_breaking_constraint(forall(i,j in C where ( i>j ))(
%               ((L[i] >= L[j]) -> n_obj[i] >= n_obj[j]) 
%             /\((L[i] < L[j]) -> n_obj[i] <= n_obj[j]) ));

constraint among(n,ass,I);

constraint alldifferent_except(ass,(n+1)..(n+1));

constraint forall(i in C)(ass[1,i]==ass[dim,i]);

constraint forall(i in C)( 
              sum(j in 2..(dim-1) where ass[j,i]!=(n+1))(S[ass[j,i]]) <= L[i]);
              
constraint forall(i in C)( 
              forall(j in 2..(dim-1))(
                ass[j,i]==(n+1) -> ass[j+1,i]==(n+1)));
                
function var int: dist_sum(array[dimension,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) =
        let 
        {       
              var int: distance= sum(i in 1..(dim-1) )(D[ass[i,courier],ass[i+1,courier]]);
        } in distance;                

constraint forall(i in C) ( dist_sum(ass,D,i) == distance[i]); 

constraint total_dist==sum(i in C)(distance[i]);   

constraint symmetry_breaking_constraint(forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
                                           (ass[2,i] >= ass[2,j])));   

solve :: seq_search([
             int_search(array1d(ass), dom_w_deg, indomain_random),
             int_search(distance, dom_w_deg, indomain_random),             
             ])
%      :: restart_luby(40)
%     minimize total_dist
      satisfy
      