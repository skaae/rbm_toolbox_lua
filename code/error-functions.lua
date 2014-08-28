require 'nn'
require 'optim'

--nInputSamples = 2
--nOutputSamples = 1
--nPoints = 10
--kernelSize = 5
--kernelStride = 1

--input = torch.rand(1, 5)

--model = nn.Linear(5,3)
--output = model:forward(input)


--conf = optim.ConfusionMatrix( {'cat','dog','person'} ) -- new matrix
--conf:zero() -- reset matrix


---- Two true positives
--conf:add( torch.Tensor({0,0,1}), 3 )
--conf:add( torch.Tensor({0,0,1}), 3 )

---- 4 True negatives
--conf:add( torch.Tensor({1,0,0}), 2 )
--conf:add( torch.Tensor({1,0,0}), 1 )
--conf:add( torch.Tensor({0,2,0}), 2 )
--conf:add( torch.Tensor({0,2,0}), 1 )
--conf:add( torch.Tensor({1,0,0}), 1 )

---- False negative
--conf:add( torch.Tensor({0,0,1}), 1 )
--conf:add( torch.Tensor({0,0,1}), 2 )


---- False Positive
--conf:add( torch.Tensor({0,1,0}), 3 )
--conf:add( torch.Tensor({1,0,0}), 3 )
--print(matthewcorr(conf))

function isNaN(number)
  return number ~= number
end

function remNaN(x,self)
      for i = 1, self.nclasses  do
      if isNaN(x[{1,i}]) then
         x[{1,i}] = 0
      end
   end
   return x
end


function getErrors(self) 
   -- returns True Posivies, false negatives, false positives, and true negatives
   local tp, fn, fp, tn
   tp  = torch.diag(self.mat):resize(1,self.nclasses )
   fn = (torch.sum(self.mat,2)-torch.diag(self.mat)):t()
   fp = torch.sum(self.mat,1)-torch.diag(self.mat)
   tn  = torch.Tensor(1,self.nclasses):fill(torch.sum(self.mat)):typeAs(tp) - tp - fn - fp

    return tp, tn, fp, fn
end


function matthewcorr(conf)
   local mcc,numerator, denominator
   tp, tn, fp, fn = getErrors(conf) 
   numerator = torch.cmul(tp,tn) - torch.cmul(fp,fn)
   denominator = torch.sqrt((tp+fp):cmul(tp+fn):cmul(tn+fp):cmul(tn+fn))
   mcc = torch.cdiv(numerator,denominator)
   mcc = remNaN(mcc,conf)
   return mcc
end

function mccerror(rbm,x,y_true)
   -- MCC error function. Returns the MCC for each class. Create a wrapper
   -- Around the function, e.g to optimize class 2 use the following error
   -- function:
   --       
   --       function errfunc(rbm,x,y_true) 
   --          local mcc  =  matthewcorr(rbm,x,y_true)
   --          return 1-mcc[{1,2}] 
   --       end
   --

     local pred,mcc,conf,_
     
     -- Convert labels given as numbers to one of K
     if y_true:size(2) ~= 1 then -- try max
          _,y_true = torch.max(y_true,2)
          y_true = y_true:typeAs(x)
     end
     
     pred = predict(rbm,x)
     
     -- Create confusion matrix
     conf = optim.ConfusionMatrix(rbm.n_classes)
     conf:batchAdd(pred,y_true)

     mcc = matthewcorr(conf)
     return mcc
end





-- print("------------------------------")
-- print('Testing statistics functions')

-- function checkequality(t1,t2,prec,pr)
--      if pr then
--           print(t1)
--           print(t2)
--      end 
--      local prec = prec or -4
--      assert(torch.numel(t1)==torch.numel(t1))
     
--      local res = torch.add(t1,-t2):abs()
--      res = torch.le(res, math.pow(10,prec))
--      res = res:sum()
--      local ret
--      if torch.numel(t1) == res then
--           ret = true
--      else
--           ret = false
--      end

--      return ret

-- end

-- function numeq(a,b)
-- 	return math.abs(a-b) < math.pow(10,-3)
-- end

--  classes = {'A', 'B','C'}
--  c = optim.ConfusionMatrix(classes)



--  -- ADD some examples
--  AC = 2; BC = 2; CC  = 5; CA = 6; AA = 7

--  c.mat[{3,1}] = AC   					--    AC = predict A true C 
--  c.mat[{3,2}] = BC  					--    BC = predict B true C 
--  c.mat[{3,3}] = CC   					--    CC = predict C true C 

--   -- ADD 7 FP for class C
--   c.mat[{1,3}] = CA  					--    CA = Predict C true A


--    -- ADD 7 TP for class A 
--    c.mat[{1,1}] = AA                    --	  AA = predict A true A 

-- tp, tn, fp, fn=getErrors(c)

-- print(tp, tn,fp,fn)

-- -- Check statistics  / check getErrors function
-- fp_test = torch.Tensor({AC,BC,CA}):resize(1,3):float()
-- fn_test = torch.Tensor({CA,0,AC+BC}):resize(1,3):float()
-- tp_test = torch.Tensor({AA,0,CC}):resize(1,3):float()
-- tn_test = torch.Tensor({BC+CC,AC+CC+CA+AA,AA}):resize(1,3):float()


-- checkequality(tp,tp_test,-10)
-- checkequality(tn,tn_test,-10)
-- checkequality(fp,fp_test,-10)
-- checkequality(fn,fn_test,-10)

-- -- -- MCC
-- mcc = matthewcorr(c)
-- test_num = tp[{1,3}] *tn[{1,3}] - fp[{1,3}] * fn[{1,3}] 
-- test_denom = math.sqrt(
-- 		     (tp[{1,3}] + fp[{1,3}])*(tp[{1,3}] + fn[{1,3}]) *
-- 			 (tn[{1,3}] + fp[{1,3}])*(tn[{1,3}] + fn[{1,3}])
-- 			 )
-- C_mcc  = test_num / test_denom
-- assert(numeq(C_mcc,mcc[{1,3}]))
-- print(C_mcc,mcc)

-- print('OK')
-- print("------------------------------")












