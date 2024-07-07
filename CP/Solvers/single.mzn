% Use this editor as a MiniZinc scratch book
% Use this editor as a MiniZinc scratch book
% Use this editor as a MiniZinc scratch book
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
%
%var 300..1200 : total_dist;
array[C] of var 0..sum(S):weights=[sum(j in dimension where ass[j,courier]!=0 ) (S[ass[j,courier]])|courier in C];
array[dimension,C] of var 0..n:ass;
array[C] of var int : Start=[ass[1,courier]| courier in C];
array[C] of var int: Finish=[ ass[(count(i in dimension)(ass[i,courier]!=0)),courier]|courier in C];
array[C] of var int: num_assigned=[(count(i in dimension)(ass[i,courier]!=0)) | courier in C];
%array[I] of var int:courier_assigned;
%constraint forall(courier in C ) ( forall( j in dimension where ass[j,courier]!=0) ( courier_assigned[ass[j,courier]]==courier));
%output[show(courier_assigned)];
constraint forall(courier in C) 
           (   
            ( forall(j,z in dimension where ass[j,courier]==0 /\ z>j) ( ass[z,courier]==0))
           );

array[C] of var int: distance=[ sum( i in 1..num_assigned[courier]-1) (D[ass[i,courier],ass[i+1,courier]])+D[n+1,Start[courier]] + 
              D[Finish[courier],n+1] |courier in C]; 
 predicate bin_packing(int: c,array[int] of var int: bin,array[int] of var int: w) =
   forall( b in lb_array(bin)..ub_array(bin) ) (
     c >= sum ( i in index_set(bin) ) (
       w[i] %* bool2int( bin[i] = b )
     )   );
%constraint bin_packing_capa(L,courier_assigned,S);
constraint forall(courier in C)(
             bin_packing(L[courier],ass[dimension,courier],[if ass[i,courier]!=0 then S[ass[i,courier]] else 0 endif|i in dimension]));
constraint forall(courier in C) (ass[1,courier]!=0);
constraint forall(courier in C) 
           (   
            ( forall(j,z in dimension where ass[j,courier]==0 /\ z>j) ( ass[z,courier]==0))
           );



%constraint forall(i in I) ( count(ass,i,1));
 constraint among(n,ass,I)::domain;
 constraint alldifferent_except_0(ass)::domain;


  %constraint total_dist==sum(i in C)(distance[i]);
%   constraint symmetry_breaking_constraint (forall(courier1,courier2 in C where(  courier1<courier2/\              
%                     L[courier1]==L[courier2]))( 
               
%                     ( ass[1,courier1]>ass[1,courier2]) )      
%                     );


int:low_max = max([D[n+1,i] +D[i,n+1] | i in 1..n]);
int: high = max(D)*((n div m)+1) + max(D[n+1,I_ext])+max(D[I_ext,n+1]);
var low_max..high: max_cap_dist; 
constraint max_cap_dist = max(distance);           
%var int: distancess=sum(i in distance) ( i);
solve :: seq_search([
          
                 int_search(ass[1..((n div m)+1)div 2,C],first_fail,indomain_random),
                %int_search(ass[(((n div m)+1)div 2)+1..((n div m)+1),C],first_fail,indomain_split)
           %  int_search(ass[1,C], dom_w_deg, indomain_random),
            %int_search(ass, dom_w_deg,indomain_random),
           
             %int_search(distance, dom_w_deg, indomain_random),
            ] )
     :: relax_and_reconstruct(array1d(ass[dimension,C]),80) 
     :: restart_luby(n)
   % ::restart_constant(70)
     % satisfy;
      minimize max_cap_dist ;

