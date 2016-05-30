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


epsilonw      = 0.1;   % Learning rate for weights
epsilonvb     = 0.1;   % Learning rate for biases of visible units
epsilonhb     = 0.1;   % Learning rate for biases of hidden units
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;


double errsum1=[];


[numcases numdims numbatches]=size(batchdata);



if restart ==1,
    restart=0;
    epoch=1;
    
    % Initializing symmetric weights and biases.
  
    vishid     = 0.1*randn(numdims, numhid);  
    labtop     = 0.1*randn(10,numhid);       
    hidbiases  = zeros(numcases,numhid);      
    visbiases  = zeros(numcases,numdims);     
    labbiases  = zeros(numcases,10);          
    
    poshidprobs = zeros(numcases,numhid);
    neghidprobs = zeros(numcases,numhid);
    posprods    = zeros(numdims,numhid);
    negprods    = zeros(numdims,numhid);
    vishidinc   = zeros(numdims,numhid);
    labtopinc   = zeros(10,numhid);
    hidbiasinc  = zeros(numcases,numhid);
    visbiasinc  = zeros(numcases,numdims);
    labbiasesinc=zeros(numcases,10);
    batchposhidprobs=zeros(numcases,numhid,numbatches);
end

for epoch = epoch:maxepoch,
    fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    for batch = 1:numbatches,
       fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);  
        target = batchtargets(:,:,batch);  
        poshidprobs = 1./(1 + exp(-data*vishid -target*labtop - hidbiases));   
        batchposhidprobs(:,:,batch)=poshidprobs;
        posprods    = data' * poshidprobs;
        poslabtopstatistics=target' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact   = sum(data);
        poslabact   = sum(target);
        
        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);  
        
        
        %%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        visible_prob_pre=1./(1+exp(-poshidstates*vishid'-visbiases));  
        
        temp_exponential=exp(poshidstates*labtop'+labbiases);
        visible_prob_post=temp_exponential./repmat(sum(temp_exponential,2),1,10);  
  
        [n_samples,n_classes] = size(visible_prob_post);
        visible_prob_post_states = zeros(n_samples,n_classes);
        r = rand(n_samples,1);
        for i = 1:n_samples
            aux = 0;
            for j = 1:n_classes
                aux = aux + visible_prob_post(i,j);
                if aux >= r(i)
                    visible_prob_post_states(i,j) = 1;
                    break;
                end
            end
        end
        
        
        
        
        neghidprobs = 1./(1 + exp(-visible_prob_pre*vishid-visible_prob_post_states*labtop - hidbiases));
        neghidstates=neghidprobs>rand(numcases,numhid);
        negprods  = visible_prob_pre'*neghidprobs;
        neglabtopstatistics=double(visible_prob_post_states') * neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(visible_prob_pre);
        neglabact=sum(visible_prob_post_states);
        
        
        
        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        err= sum(sum( (data-visible_prob_pre).^2 ));
        errsum = err + errsum;
        errsum1(epoch)=errsum;
        
        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        %%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        vishidinc = momentum*vishidinc + ...
            epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(repmat(posvisact,numcases,1)-repmat(negvisact,numcases,1));
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(repmat(poshidact,numcases,1)-repmat(neghidact,numcases,1));
        labtopinc=momentum * labtopinc + epsilonw * ((poslabtopstatistics-neglabtopstatistics)/numcases- weightcost * labtop);
        labbiasesinc=momentum *labbiasesinc + (epsilonhb/numcases)*(repmat(poslabact,numcases,1)-repmat(neglabact,numcases,1));
        
        vishid = vishid + vishidinc;
        labtop=labtop+labtopinc;
        visbiases = visbiases + visbiasinc;
        hidbiases = hidbiases + hidbiasinc;
        labbiases=labbiases+labbiasesinc;
        
        
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
end;

save rbmerr errsum1;
