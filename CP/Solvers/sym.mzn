include "globals.mzn";

int:m;
int:n;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions


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
            

function var int: dist_sum(array[I_ext,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) = sum(i in I_ext )(D[i,ass[i,courier]] );                                                                           


array[C] of var int:distance=[dist_sum(next_obj,D,courier)| courier in C];


constraint forall( i in C)(next_obj[last_obj[i],i] == n+1);

 predicate isSym(array[I_ext] of var I_ext: arr, array[I_ext,I_ext] of int: D) =
     forall( i in I_ext) (
         D[i,arr[i]] == D[arr[i],i]);

 constraint symmetry_breaking_constraint(forall(i in C)(
              isSym(next_obj[I_ext,i],D) -> (next_obj[n+1,i]<= last_obj[i]) ));
             
 constraint symmetry_breaking_constraint(forall(i in C)(
             (next_obj[n+1,i]<= last_obj[i])-> (isSym(next_obj[I_ext,i],D) -> (next_obj[n+1,i]<= last_obj[i])) ));             
             

 constraint symmetry_breaking_constraint(forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
                                            next_obj[n+1,i] >= next_obj[n+1,j]));




%%%%%%%%%%%%%%%%%%%%%%%%Second model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int:dim = (n div m)+3;
set of int:dimension=1..dim;
array[dimension,C] of var I_ext:ass; 

constraint among(n,ass,I)::domain;
constraint alldifferent_except(ass,(n+1)..(n+1))::domain;


constraint all_equal(ass[1,C]++ass[dim,C]); %First and last row are the same

array[C] of var int :n_obj = [(count(j in dimension )(ass[j,courier]!=n+1 )) | courier in C];

predicate bin_packing(int: c,array[int] of var int: bin,array[int] of var int: w) =
  forall( b in lb_array(bin)..ub_array(bin) ) (
    c >= sum ( i in index_set(bin) ) (
      w[i]
    )
  );
array[C] of var int: weights= [sum(i in dimension where ass[i,courier]!=n+1) (S[ass[i,courier]])|courier in C];
  constraint symmetry_breaking_constraint( forall(courier1,courier2 in C where courier1<courier2 /\ L[courier1]>=weights[courier2] /\ L[courier2]>= weights[courier1]) ( ass[2,courier1]< ass[2,courier2]));

constraint forall(courier in C)(
              bin_packing(L[courier],ass[dimension,courier],[if ass[i,courier]!=n+1 then S[ass[i,courier]] else 0 endif|i in dimension]));
  

constraint forall(i in C, j in 1..(dim-n_obj[i]-2))( 
                ass[dim-j,i]==(n+1));                
                                                


%%%%%%%%%%%%%%%%%%%%%%%%Model Channelling%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

constraint forall(i in C,j in 1..(n_obj[i]+1))(
              next_obj[ass[j,i],i] == ass[j+1, i]);

%%%%%%%%%%%%%%%%%%%%%%%%Search Heuristic%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  int:low_max = max([D[n+1,i] +D[i,n+1] | i in 1..n]);
int: high = sum(i in 1..(n div m +1))(D[i,i+1]) + low_max;
var low_max..high: max_cap_dist; 
constraint max_cap_dist = max(distance);           

solve :: seq_search([
          
             
          
             int_search(ass[dimension,i],dom_w_deg,indomain_random)|i in C
          ] )
  
     :: relax_and_reconstruct(array1d(ass),80) 
     :: restart_luby(n)
  
      minimize max_cap_dist::domain ;

