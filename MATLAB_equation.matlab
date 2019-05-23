function [pexpr, rhsexpr, problem, PtrsCreated] = solve(d,exprptr, ptr)
numptrfound = 0;
inputs = getinputs(d);
top = inputs(1:d.NTop);if d.NBottom
    bottom = inputs(d.NTop+1:end);end
topnop = [];for i=1:d.NTop
    if top(i)==ptr 
       found = 'isolatedtop';
       pexpr = top(i).info; 
       numptrfound = numptrfound + 1; 
    else
       t_ptrs = top(i).getptrs;
       if isempty(t_ptrs) || ~any(t_ptrs == ptr) 
           topnop = [topnop top(i)];
       else 
           found = 'buriedtop';
           pexpr = top(i).info;
           numptrfound = numptrfound + 1; 
       end
    endend
bottomnop = [];for i=1:d.NBottom
    if  bottom(i)==ptr 
        found = 'isolatedbottom';
        pexpr = bottom(i).info;
        numptrfound = numptrfound + 1; 
    else
        b_ptrs = bottom(i).getptrs;
        if isempty(b_ptrs) || ~any(b_ptrs == ptr)
            bottomnop = [bottomnop bottom(i)];
        else
            found = 'buriedbottom';
            pexpr = bottom(i).info;
            numptrfound = numptrfound + 1;
        end
    endend
if isequal(numptrfound,0) 
   problem = 'The pointer does not appear in the divide expression.';
   rhsexpr = exprptr;
   pexpr = [];
   PtrsCreated = [];
   returnelseif numptrfound > 1;
   problem = 'The pointer appears more that once in the expression.';
   rhsexpr = exprptr;
   pexpr = d;
   PtrsCreated = [];
   returnend

switch foundcase 'isolatedtop'
   rhsexpr = xregpointer(cgdivexpr('rhsexpr',[exprptr bottomnop],topnop));
   PtrsCreated = rhsexpr;
   problem = 0;case 'isolatedbottom'
   rhsexpr = xregpointer(cgdivexpr('rhsexpr',topnop,[exprptr bottomnop]));
   PtrsCreated = rhsexpr;
   problem = 0;case 'buriedtop'
   rhsexprtmp = xregpointer(cgdivexpr('rhsexpr',[exprptr bottomnop],topnop));
   % dig deeper
   [pexpr,rhsexpr,problem, PtrsCreated] = solve(pexpr,rhsexprtmp,ptr);
   PtrsCreated = [PtrsCreated rhsexprtmp];case 'buriedbottom'
   % form rhsexpr =  topnop/(bottomnop*cgexpr) 
   rhsexprtmp = xregpointer(cgdivexpr('rhsexpr',topnop,[exprptr bottomnop]));
   % dig deeper
   [pexpr,rhsexpr,problem, PtrsCreated] = solve(pexpr,rhsexprtmp,ptr);
   PtrsCreated = [PtrsCreated rhsexprtmp];end