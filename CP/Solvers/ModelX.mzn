include "globals.mzn";

int:m;
int:n;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions

int:low = min([D[n+1,i] +D[i,n+1] | i in 1..n]);
int: high = sum(i in 1..(n div m +1))(D[i,i+1]) + low;
var int:UB;
var int:LB ;
constraint UB = high;
constraint LB = low;
array[C] of var low..high: distance;


array[I_ext, C] of var I_ext: next_obj; 
% array[C] of var I: last_obj;

constraint forall(i in C)(
              subcircuit(next_obj[I_ext,i])
            );
constraint forall(i in C)(
              count(next_obj[I,i],n+1)==1
            );                 
   
constraint forall(k in C) (
                  next_obj[n+1,k] != n+1   
            );            
                              
constraint forall(i in I)(
              among(1,next_obj[i,C],I_ext diff {i})
            );
            

function var int: dist_sum(array[I_ext,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = sum(i in I_ext )(D[i,ass[i,courier]] );                                                                           


constraint forall(i in C) ( dist_sum(next_obj,D,i) == distance[i]); 

% constraint forall(k in C) (
%       distance[k] = sum([D[i, next_obj[i, k]]| i in I_ext where next_obj[i, k] != i])
% );

% constraint forall( i in C)(next_obj[last_obj[i],i] == n+1);

% predicate isSym(array[I_ext] of var I_ext: arr, array[I_ext,I_ext] of int: D) =
%     forall( i in I_ext) (
%         D[i,arr[i]] == D[arr[i],i]);

% constraint symmetry_breaking_constraint(forall(i in C)(
%              isSym(next_obj[I_ext,i],D) -> (next_obj[n+1,i]<= last_obj[i]) ));
             
% constraint symmetry_breaking_constraint(forall(i in C)(
%             (next_obj[n+1,i]<= last_obj[i])-> (isSym(next_obj[I_ext,i],D) -> (next_obj[n+1,i]<= last_obj[i])) ));             
             

% constraint symmetry_breaking_constraint(forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
%                                            next_obj[n+1,i] >= next_obj[n+1,j]));

% int:max_obj = n div m + 1;
% array[C] of var (max_obj-1)..max_obj:n_obj;
array[C] of var int:n_obj;
constraint forall(i in C) ( n_obj[i] == (sum(j in I_ext where  next_obj[j,i]!=j )(1)-1)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
array[I]of var C:obj_ass;
constraint forall(i in C,j in I where next_obj[j,i]!=j)(obj_ass[j]=i);
constraint bin_packing_capa(L,obj_ass, S);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% int :max_load = max(L) + 1;
% array[C] of var 0..max_load: couriers_loads;
% constraint forall(k in C)(
%     couriers_loads[k] = sum(
%             [S[i] | i in I where obj_ass[i]=k]
%     )
% );

int:low_max = max([D[n+1,i] +D[i,n+1] | i in 1..n]);
var int:LOW_DOM ;
constraint LOW_DOM = low_max;
var low_max..high: max_cap_dist; 
constraint max_cap_dist == max(distance);

solve :: seq_search([
%              int_search(next_obj, first_fail, indomain_min),
            int_search(next_obj, dom_w_deg, indomain_random ),             
             ])
       :: restart_luby(100)
     minimize max_cap_dist;
     