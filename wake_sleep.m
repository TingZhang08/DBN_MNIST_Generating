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

maxepoch=10000;
numhid=500; numpen=500; numpen2=2000;



labgenbiases=zeros(numcases,10);
pengenbiases=zeros(numcases,numpen);
wakehidstates=zeros(numcases,numhid);
wakepenstates=zeros(numcases,numpen);
postopstates=zeros(numcases,numpen2);
neglabstates=zeros(numcases,10);
penhid=hidpen';


epsilonw = 0.01;   
epsilonvb = 0.01; 
epsilonhb = 0.01; 
weightcost  = 0.0002;
initialmomentum  = 0.5;
finalmomentum    = 0.9;



display('Generatively fine-tuning the model using wake sleep........');
load mnistvhclassify
load mnisthpclassify
load mnisthp2classify

makebatches;
[numcases numdims numbatches]=size(batchdata);
N=numcases;


hidvis=vishid';
hidvisinc=zeros(numhid,numdims);
visbiasinc=zeros(numcases,numdims);
penhidinc=zeros(numpen,numhid);
hidgenbiasesinc=zeros(numcases,numhid);
labtopinc=zeros(10,numpen2);
labgenbiasesinc=zeros(numcases,10);
hidpen2inc=zeros(numpen,numpen2);
pengenbiasesinc=zeros(numcases,numpen);
penrecbiases2inc=zeros(numcases,numpen2);
hidpeninc=zeros(numhid,numpen);
penrecbiasesinc=zeros(numcases,numpen);
vishidinc=zeros(numdims,numhid);
hidrecbiasesinc=zeros(numcases,numhid);



  

for epoch = 1:maxepoch
    for batch = 1:numbatches
        data = [batchdata(:,:,batch)];  
        target = [batchtargets(:,:,batch)];  
        wakehidprobs= 1./(1 + exp(-data*vishid-hidrecbiases)); 
        wakehidstates=wakehidprobs>rand(numcases,numhid);
        wakepenprobs=1./(1+exp(-wakehidstates*hidpen-penrecbiases));  
        wakepenstates=wakepenprobs>rand(numcases,numpen);
        postopprobs=1./(1+exp(-wakepenstates*hidpen2-target*labtop-penrecbiases2)); 
        postopstates=postopprobs>rand(numcases,numpen2);
       
		 poslabtopstatistics=target' * postopprobs;
         pospentopstatistics=wakepenstates' *postopprobs;

         posvisact=sum(data);
         postarget=sum(target);
         pospengen=sum(wakepenstates);
         postop=sum(postopstates);
                 
        % perform numCDiters Gibbs Sampling iterations using the top level
        % undirected associative memory
        
            negtopstates=postopstates;
              if (1<=epoch<=100)
             numCDiters=3;
             elseif (101<=epoch<=200)
            umCDiters=6;
             elseif (201<=epoch<=10000)
             umCDiters=10;
             end
            
            % the top level RBM
            for iter=1:numCDiters
              
                neglabstates=zeros(numcases,10);
                
                negpenprobs=1./(1+exp(-negtopstates*hidpen2'-pengenbiases));
                negpenstates=negpenprobs>rand(numcases,numpen);
                neglabprobs=exp(negtopstates*labtop'+labgenbiases);
                neglabprobs=neglabprobs./(repmat(sum(neglabprobs,2),1,10));

               % sample y
               [n_samples,n_classes] = size(visible_prob_post);
               neglabstates = zeros(n_samples,n_classes);
               r = rand(n_samples,1);
               for ii = 1:n_samples
                aux = 0;
                for j = 1:n_classes
                       aux = aux + visible_prob_post(ii,j);
                  if aux >= r(ii)
                    neglabstates(ii,j) = 1;
                    break;
                  end
                end
               end

       
                negtopprobs=1./(1+exp(-negpenstates*hidpen2-neglabstates*labtop-penrecbiases2));
                negtopstates=negtopprobs>rand(numcases,numpen2);
                
            end
       

		
		negpentopstatistics=double(negpenstates') *double( negtopprobs);
        neglabtopstatistics=double(neglabstates') * negtopprobs;
        
        negtarget=sum(neglabstates);
        negtop=sum(negtopstates);
        
          
        
        % starting from the end of the Gibbs SAMPLING RUN, perform a
        % top-down generative pass to get sleep phase probabilities and
        % sample states
        
        sleeppenstates=negpenstates;
        sleephidprobs=1./(1+exp(-sleeppenstates*penhid-hidgenbiases));
        sleephidstates=sleephidprobs>rand(numcases,numhid);
        sleepvisprobs=1./(1+exp(-sleephidstates*hidvis-visbiases));
        
        % predictions
        psleeppenstates=1./(1+exp(-sleephidstates*hidpen-penrecbiases));  
		
		possleeppenstates=psleeppenstates>rand(numcases,numpen);
		
        psleephidstates=1./(1+exp(-sleepvisprobs*vishid-hidrecbiases));     
		
		negsleephidstates=psleephidstates>rand(numcases,numhid);
		
        pvisprobs=1./(1+exp(-wakehidstates*hidvis-visbiases));             
        phidprobs=1./(1+exp(-wakepenstates*penhid-hidgenbiases));
		
		phidstates=phidprobs>rand(numcases,numhid);
        
        
        negvisact=sum(pvisprobs);
        poshidgen=sum(wakehidstates);
        neghidgen=sum(phidstates);
        negpengen=sum(negpenstates);
        possleeppen=sum(sleeppenstates);
        negpsleeppen=sum(possleeppenstates);
        possleephid=sum(sleephidstates);
        negpsleephid=sum(negsleephidstates);
        
        if epoch>5,
            momentum=finalmomentum;
        else
            momentum=initialmomentum;
        end;
        
        % update to generative parameters
 
         hidvisinc=momentum * hidvisinc + epsilonw*(wakehidprobs'*(data-pvisprobs)/numcases-weightcost * hidvis);
         visbiasinc=momentum * visbiasinc + (epsilonvb/numcases)*(repmat(posvisact,numcases,1)-repmat(negvisact,numcases,1));
         penhidinc=momentum * penhidinc + epsilonw*(wakepenprobs'*(wakehidstates-phidstates)/numcases-weightcost * penhid);
         hidgenbiasesinc=momentum * hidgenbiasesinc + (epsilonhb/numcases)*(repmat(poshidgen,numcases,1)-repmat(neghidgen,numcases,1));
    
        
        hidvis=hidvis+hidvisinc;
        visbiases=visbiases+visbiasinc;
        penhid=penhid+ penhidinc;
        hidgenbiases=hidgenbiases+hidgenbiasesinc;
        
        
        
        % update to top level associative memory parameters
        labtopinc=momentum * labtopinc + epsilonw * ((poslabtopstatistics-neglabtopstatistics)/numcases- weightcost * labtop);
        labgenbiasesinc=momentum *labgenbiasesinc + (epsilonhb/numcases) *(repmat(postarget,numcases,1)-repmat(negtarget,numcases,1));
        hidpen2inc=momentum * hidpen2inc + epsilonw * ((pospentopstatistics-negpentopstatistics)/numcases- weightcost * hidpen2);
        pengenbiasesinc=momentum * pengenbiasesinc + (epsilonhb/numcases) * (repmat(pospengen,numcases,1)-repmat(negpengen,numcases,1));
        penrecbiases2inc=momentum * penrecbiases2inc + (epsilonhb/numcases) * (repmat(postop,numcases,1)-repmat(negtop,numcases,1));
        
        
        labtop=labtop +labtopinc;
        labgenbiases=labgenbiases+labgenbiasesinc;
        hidpen2=hidpen2 + hidpen2inc;
        pengenbiases=pengenbiases+pengenbiasesinc;
        penrecbiases2=penrecbiases2 + penrecbiases2inc;
        
        
        % update to recognition approximation parameters
           
		hidpeninc=momentum * hidpeninc + epsilonw * ((sleephidprobs' * (sleeppenstates-possleeppenstates)/numcases)-weightcost * hidpen);
		vishidinc=momentum * vishidinc + epsilonw *((sleepvisprobs' * (sleephidstates-negsleephidstates)/numcases)-weightcost * vishid);
		penrecbiasesinc=momentum * penrecbiasesinc + (epsilonhb/numcases) * (repmat(possleeppen,numcases,1)-repmat(negpsleeppen,numcases,1));
        hidrecbiasesinc=momentum * hidrecbiasesinc + (epsilonhb/numcases) * (repmat(possleephid,numcases,1)-repmat(negpsleephid,numcases,1));
        
        
        hidpen=hidpen + hidpeninc;
        penrecbiases=penrecbiases+penrecbiasesinc ;
        vishid=vishid + vishidinc ;
        hidrecbiases=hidrecbiases+ hidrecbiasesinc;
         
        
    end
   
   
  
save parameters hidvis visbiases penhid hidgenbiases labtop labgenbiases hidpen2 pengenbiases penrecbiases2 hidpen penrecbiases vishid hidrecbiases;

    
    fprintf(1,'After epoch %d Train. \n',epoch);
    

      generating_samples;
  
 
end
% generating_samples;   % You can also test your network after
                              % training
