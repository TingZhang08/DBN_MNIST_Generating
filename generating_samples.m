% Code provided by Yujian Li and Ting Zhang.  

% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

load parameters;


maxepoch=1000;


data=[];
datagen=[];
testtarget=[];


testtarget = [testtarget; repmat([1 0 0 0 0 0 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 1 0 0 0 0 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 1 0 0 0 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 1 0 0 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 1 0 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 0 1 0 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 0 0 1 0 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 0 0 0 1 0 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 0 0 0 0 1 0], 5, 1)];
testtarget = [testtarget; repmat([0 0 0 0 0 0 0 0 0 1], 5, 1)];

[n_test,n_sample]=size((testtarget));


pen_sampled=double(hidgenbiases(1:n_test,:)>rand(n_test,numpen));
topprobs=1./(1+exp(-testtarget*labtop-pen_sampled*hidpen2-penrecbiases2(1:n_test,:)));
negtopstates = topprobs > rand(n_test,numpen2);

%%%%%%%%%%%%%%%%% Gibbs Samples %%%%%%%%%%%%%%%%%%%%%%%%
for epoch = 1:maxepoch
    
    negpenprobs=1./(1+exp(-negtopstates*hidpen2'-pengenbiases(1:n_test,:)));
    negpenstates=negpenprobs>rand(n_test,numpen);
    negtopprobs=1./(1+exp(-negpenstates*hidpen2-testtarget*labtop-penrecbiases2(1:n_test,:)));
    negtopstates=negtopprobs>rand(n_test,numpen2);
    
end


% Up-Bottom Phase
penprobsgen=1./(1+exp(-topprobs*hidpen2'-pengenbiases(1:n_test,:)));
hidprobsgen=1./(1+exp(-penprobsgen*penhid-hidgenbiases(1:n_test,:)));
datagen=1./(1+exp(-hidprobsgen*hidvis-visbiases(1:n_test,:)));

 figure(1);
dispims(datagen',28,28,0,2,5);







