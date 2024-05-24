include "globals.mzn";
int:m;
int:n;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :p = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[p,p] of int: D;    % Distance Distributions
int: total = sum(i in p, j in p)(D[i,j]);
int: total_max = total * m;
var 0..total_max : total_dist;

array[C] of var 1..total: distances; % First column the item 
                                    % Second column the courier 
                                    % The indexes raprent the order
array[I,1..2] of var I: obj_assign;

constraint forall(i in C)(
            sum(j in I where obj_assign[j,2]=i)( S[obj_assign[j,1]] ) <= L[i]);
            
constraint forall(i in I)(
            obj_assign[i,2] <= m);
            
constraint all_different(obj_assign[I,1]);

function var int: dist_sum(int: courier, array[I,1..2] of var int: assign, array[p, p] of int: D) =
    let {
        array[int] of var opt int: filtered_items = [0]++
                                                    [assign[i,1] | i in 1..n where assign[i,2] == courier]++
                                                    [0];
                                                 
        var int : distance = sum(index in 1..(n-1)
                              where ( filtered_items[index] != <> /\ filtered_items[next_index] != <>))
                                (filtered_items[index]);
        
    } in distance;
                              
constraint forall(i in C)(
            dist_sum(i,obj_assign,D) == distances[i]);
