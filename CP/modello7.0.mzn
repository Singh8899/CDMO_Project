include "globals.mzn";
int:m;
int:n;

set of int:dimension=1..(n div m)+1;
set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions
%var int :max_distance= max(distance);
%var int :min_distance= min(distance);
array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions
int: total = sum(i in I_ext, j in I_ext)(D[i,j]);
int: total_max = total * m;
var 0..total_max : total_dist;
array[C] of var 0..sum(S):weights;
array[I] of var C:item_ass;
array[I] of var I:item_order;
array[I] of int: low_bounds = [D[n+1,i] +D[i,n+1] | i in 1..n] ;
int:low = min(low_bounds);
array[C] of var I:Start;
array[C] of var I:Finish;
constraint forall(courier in C) ( sum( i in I where item_ass[i] == courier) (S[i]) ==weights[courier] /\ weights[courier]<=L[courier]); 
constraint forall(courier in C) ( forall( i in I where item_ass[i]==courier /\ item_order[i]==1) (Start[courier]==i)) ;
constraint forall(courier in C) ( forall( i in I where item_ass[i]==courier /\ item_order[i]==(count(z in I) (item_ass[z]==courier)))
           (Finish[courier]==i));
int: high = sum(i in 1..(n-1-m))(D[i,i+1]) + D[n-m+1,1] + D[n-m,n+1];
array[C] of var low..high: distance; 
% output[show(item_ass)];
% output[show(item_order)];
% output[show(Finish)];
% output[show((count(z in I) (item_ass[z]==1)))];
constraint forall(courier in C) (count(i in I)(item_ass[i]==courier) >=2);
constraint forall(courier in C) ( forall(order in 1.. (count(i in I) (item_ass[i]==courier)) ) ( count( j in I where item_ass[j]==courier) ( item_order[j]==order)==1));

function var int: dist_sum(array[I] of var C:item_ass,array[I_ext, I_ext] of int: D,int :courier) =
        let 
        {       
              var int: distance= sum( i,j in I where item_ass[i] == courier /\ item_ass[j]==courier /\ item_order[i]-item_order[j]==-1)(D[i,j])
              +D[n+1,Start[courier]] + D[Finish[courier],n+1]
         } in distance;
%constraint forall(courier1,courier2 in C where L[courier1]==L[courier2] /\ courier1>courier2) (  
constraint forall(courier in C) (distance[courier]==dist_sum(item_ass,D,courier));
constraint total_dist==sum(i in C)(distance[i]);
solve minimize total_dist;