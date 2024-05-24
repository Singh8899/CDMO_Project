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

int: high = sum(i in 1..(dim-2))(D[i,i+1]) + D[n-m+1,1] + D[n-m,n+1];

array[C] of var low..high: distance;
var int:UB;
var int:LB ;
constraint UB = high;
constraint LB = low;

%%%%%%%%%%%%%%%%%%%%%%%%First model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

array[I_ext, C] of var I_ext: next_obj; 
array[C] of var I: last_obj;


constraint forall(i in C)(
              subcircuit(next_obj[I_ext,i])
            /\(count(next_obj[I,i],n+1)==1)
            )::domain;

constraint forall(i in I)(
              among(1,next_obj[i,C],I_ext diff {i})
            )::domain;
            
% constraint forall(i in C)( 
%               sum(j in I_ext where next_obj[j,i]!=j /\ next_obj[j,i]!=(n+1))(S[next_obj[j,i]]) <= L[i]);

function var int: dist_sum(array[I_ext,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = sum(i in I_ext )(D[i,ass[i,courier]] );                                                                           


constraint forall(i in C) ( dist_sum(next_obj,D,i) == distance[i]); 

constraint forall( i in C)(next_obj[last_obj[i],i] == n+1)::domain;

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


%%%%%%%%%%%%%%%%%%%%%%%%Second model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int:dim = (n div m)+1+2;
set of int:dimension=1..dim;
array[dimension,C] of var I_ext:ass; 

constraint alldifferent_except(ass,(n+1)..(n+1))::domain;
constraint all_equal(ass[1,C]++ass[dim,C])::domain; %First and last row are the same


array[I]of var C:obj_ass;
constraint forall(i in C,j in I where next_obj[j,i]!=j)(obj_ass[j]=i);
constraint bin_packing_capa(L,obj_ass, S);


% constraint forall(i in C)( 
%               sum(j in 2..(dim-1) where ass[j,i]!=(n+1))(S[ass[j,i]]) <= L[i]);


constraint forall(i in C, j in 1..(dim-n_obj[i]-2))( 
                ass[dim-j,i]==(n+1));     

                               

% function var int: rev_dist_sum(array[dimension,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = 
%       let
%       {
%         array[dimension] of var I_ext:rev_arr = reverse(ass[..,courier]);
%       }
%       in sum(i in 1..(dim-1) )(D[rev_arr[i],rev_arr[i+1]]);                                


% constraint forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
%                                            ass[2,i] >= ass[2,j]);
                                           
% constraint symmetry_breaking_constraint(forall(i in C )(      
%                                            distance[i] <= rev_dist_sum(ass,D,i )));                                             
                                           
%%%%%%%%%%%%%%%%%%%%%%%%Model Channelling%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constraint forall(i in C,j in 1..(n_obj[i]+1))(
              next_obj[ass[j, i],i] = ass[j+1, i]);

%%%%%%%%%%%%%%%%%%%%%%%%Search Heuristic%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solve :: seq_search([
%              int_search(next_obj, first_fail, indomain_min),
              int_search(next_obj, dom_w_deg, indomain_random ),
             ])
       :: restart_luby(100)
%     satisfy
      minimize max(distance);