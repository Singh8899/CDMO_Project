include "globals.mzn";
% PARAMETERS 
int:m;
int:n;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions

int: total_max = sum(j in I_ext, i in 1..2)(D[i,j]);
var 0..total_max : total_dist;

%array[I] of int: low_bounds = [D[n+1,i] +D[i,n+1] | i in 1..n] ;
% int:low = min(low_bounds);

% int: high = sum(i in 1..((n div m)+1))(D[i,i+1]) + D[n-m+1,1] + D[n-m,n+1];
% array[C] of var low..high: distance;

% var int:UB;
% var int:LB ;
% constraint UB = high;
% constraint LB = low;
array[C] of var 1..(dim-2) :n_obj;

constraint sum(i in n_obj)(i) <= n;
constraint forall(i in C)(
              n_obj[i] = sum(j in 2..(dim-1) )(ass[j,i] != (n+1)));

predicate isSym(array[dimension] of var I_ext: arr, array[I_ext,I_ext] of int: D) =
    forall( i in 1..((n div m)+2)) (
        D[arr[i],arr[i+1]] == D[arr[i+1],arr[i]]);

constraint symmetry_breaking_constraint(forall(i in C)(
             isSym(ass[dimension,i],D) -> (ass[2,i]<= ass[n_obj[i]+1,i]) ));
     
int:dim = (n div m)+1+2;
set of int:dimension=1..dim;
array[dimension,C] of var I_ext:ass; 

constraint among(n,ass,I);
constraint alldifferent_except(ass,(n+1)..(n+1));



constraint all_equal(ass[1,C]++ass[dim,C]); %First and last row are the same

constraint forall(i in C)( 
              sum(j in 2..(dim-1) where ass[j,i]!=(n+1))(S[ass[j,i]]) <= L[i]);

constraint forall(i in C, j in 1..(dim-n_obj[i]-2))( 
                ass[dim-j,i]==(n+1));                
                                                
function var int: dist_sum(array[dimension,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = sum(i in 1..(dim-1) )(D[ass[i,courier],ass[i+1,courier]]);

function var int: rev_dist_sum(array[dimension,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = 
      let
      {
        array[dimension] of var I_ext:rev_arr = reverse(ass[..,courier]);
      }
      in sum(i in 1..(dim-1) )(D[rev_arr[i],rev_arr[i+1]]);                                

array[C] of var int: distance;
constraint forall(i in C) ( dist_sum(ass,D,i) == distance[i]); 
constraint total_dist==sum(i in C)(distance[i]);

constraint symmetry_breaking_constraint(forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
                                           ass[2,i] >= ass[2,j]));
                                           
constraint symmetry_breaking_constraint(forall(i in C )(      
                                           distance[i] <= rev_dist_sum(ass,D,i )));                                             


% array[C,dimension] of var I_ext: rot_ass; 

% constraint forall(j in dimension,i in C )(rot_ass[i,j] == ass[j,i]);

solve :: seq_search([
             int_search(ass, dom_w_deg, indomain_random ),
%             int_search(distance, dom_w_deg, indomain_random),             
             ])
%      :: restart_luby(50)
     satisfy
%      minimize max(distance);
%      minimize total_dist ;
%output [show(array1d(rot_ass)[i]) | i in 1..10];
%solve minimize total_dist; 
