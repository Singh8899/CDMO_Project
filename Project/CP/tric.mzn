include "globals.mzn";
include "lex_lesseq.mzn";

%-----------------------------------------------------------------------------%
% Parameters
%-----------------------------------------------------------------------------%

int: m; % couriers
set of int: COURIERS = 1..m;
array[COURIERS] of int: l;

int: n; % items
set of int: ITEMS = 1..n;
array[ITEMS] of int: s; % items sizes

set of int: D_SIZE = 1..n+1;
array[D_SIZE, D_SIZE] of int: D; % distances

%-----------------------------------------------------------------------------%
% Variables
%-----------------------------------------------------------------------------%

% order of items for each courier
array[COURIERS, D_SIZE] of var D_SIZE: T;


%-----------------------------------------------------------------------------%
% Constraints
%-----------------------------------------------------------------------------%

constraint 
    forall(i in COURIERS) (
        sum(j in ITEMS where T[i,j] != j)(s[j]) <= l[i]    % load capacities are respected
        /\ ((count(j in ITEMS)(T[i,j] != j) > 0) -> (T[i,n+1] != n+1))  % if courier delivers items then the node origin must be
                                                                        % present in the subcircuit (complementary of implied c.)
    );

constraint
    forall(j in ITEMS) (
        count(z in T[..,j])(z != j) == 1 % each item is transported by exactly one courier
    );
        
% ----------COMMENT TO REMOVE SYMMETRY BREAKING----------%
% Symmetry breaking constraint: if two couriers can also deliver each other's loads -> accept only 
% one combination as valid
constraint
    forall(i,j in COURIERS where i < j)(
        (sum(k in ITEMS where T[i,k] != k)(s[k]) <= l[j] /\ sum(k in ITEMS where T[j,k] != k)(s[k]) <= l[i])
        -> lex_lesseq(T[i,..], T[j,..])
    );
%----------COMMENT TO REMOVE SYMMETRY BREAKING----------%

% constraints to create T as a set of Hamiltonian sub-cycles (or sub-circuits)
constraint 
    forall(i in COURIERS) (
        subcircuit(T[i,..])
    );


%-----------------------------------------------------------------------------%
% Objective
%-----------------------------------------------------------------------------%

int: obj_lowerbound = max(i in ITEMS)(D[n+1,i] + D[i,n+1]);

array[ITEMS] of int: max_dists = sort([max(j in ITEMS)(D[i,j]) | i in ITEMS]);
int: obj_upperbound = sum(i in 2..n)(max_dists[i]) + max(j in ITEMS)(D[n+1,j]) + max(j in ITEMS)(D[j,n+1]);
        
var obj_lowerbound..obj_upperbound: obj = max(i in COURIERS)(sum(j in D_SIZE where T[i,j] != j) (D[j,T[i,j]]));


%-----------------------------------------------------------------------------%
% Search Strategy
%-----------------------------------------------------------------------------%
solve :: int_search(T, dom_w_deg, indomain_min) 
         minimize obj;

%-----------------------------------------------------------------------------%
% Output
%-----------------------------------------------------------------------------%

output  [show(obj) ++ "\n"] ++
        [show(T[i,j]) ++ if j == n+1 then "\n" else " " endif | i in COURIERS, j in D_SIZE]