% Code to output a node based array for the oto problems
function outputindexarray(m,n,dim,index,array,name)

  narray = reshape(array,dim,m,n);
  printf("\n%s :\n",name)
  for jIdx = m:-1:1
    for iIdx = 1:n-1
      printf("%8.3f ",narray(index,jIdx,iIdx))
    end  
    printf("%8.3f\n",narray(index,jIdx,n))
  end

end
