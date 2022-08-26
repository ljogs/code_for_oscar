 

% X should be a matrix whose rows are the observations and columns are the
% predictors (n by p). The intercept is omitted (so the response and each
% predictor should have mean zero).

function [op_coef, df] = oscar_package_real_data_finial_model(tuning)

format long

X_train = readmatrix("OSCAR_X_train.csv");
y_train = readmatrix("OSCAR_y_train.csv");

cvalues = [0; 0.001; 0.01; .01; .05; .1;.25;.5;.55;.6;.7;.75;.8;.85;.9;.95;1];
propvalues = [.0001; .001; .002; .00225; .0025; .00275; .0028; .0029; .003; .004; .005; .0075; .01;.025; .05; .1; .15;.2;.25;.3;.35;.4;.45;.5;.55;.6;.65;0.7;0.75];

p = length(X_train(1,:));

[initcoef] = regress(y_train,X_train); 

Xmatrix = [X_train -X_train]; 

[initcoeford, currorder] = sort(-abs(initcoef));
sameaslast = [0; (initcoeford(2:p) == initcoeford(1:(p-1)))];
startblocksame = [((sameaslast(2:p) - sameaslast(1:(p-1))) > 0); 0];

endblocksame = [((sameaslast(2:p) - sameaslast(1:(p-1))) < 0); sameaslast(p)];
nblocksame = sum(startblocksame);
vi = (1:p)';
visbs = vi(logical(startblocksame));
viebs = vi(logical(endblocksame));
for i = 1:nblocksame; 
    blockmean = mean(vi(visbs(i):viebs(i)));
    vi(visbs(i):viebs(i)) = blockmean * ones(viebs(i) - visbs(i) + 1,1);
end;
[tempinvsort,vind] = sort(currorder);
a1 = vi(vind)';
initcoeford = -initcoeford;   

CoefMatrix = zeros(p,length(propvalues),length(cvalues));
Test_Error_Matrix = zeros(length(propvalues),length(cvalues));
SSMatrix = zeros(1,length(propvalues),length(cvalues));
dfMatrix = zeros(1,length(propvalues),length(cvalues));

for ccount = 1:length(cvalues)
    OrderMatrix = a1;
    weighting = a1;
    cvalue = cvalues(ccount);
    for i=1:p
        weighting(i) = (1-cvalue)+cvalue*(p-i);
    end;    
    for propcount = 1:length(propvalues)
        tbound = propvalues(propcount)*weighting*initcoeford;
        if (ccount == 1)
            if (propcount == 1)
                start=zeros(2*p,1);
            end;
        elseif (propcount > 1)
            start=[max(CoefMatrix(:,propcount-1,ccount),0);-min(CoefMatrix(:,propcount-1,ccount),0)];
        else
            start=[max(CoefMatrix(:,propcount,ccount-1),0);-min(CoefMatrix(:,propcount,ccount-1),0)];
        end;
        [coefs df ssquares conv] = OscarSeqOpt(tbound, cvalue, Xmatrix, y_train, start, OrderMatrix);
        CoefMatrix(:,propcount,ccount) = coefs;
        SSMatrix(:,propcount,ccount) = ssquares;
        dfMatrix(:,propcount,ccount) = df;
    end;
end;

op_coef=CoefMatrix(:,tuning(1), tuning(2));

