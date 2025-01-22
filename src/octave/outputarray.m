% Code to output a node based array for the oto problems
function outputarray(m,n,array,name)

  narray = reshape(array,m,n);
  printf("\n%s :\n",name)
  for jIdx = m:-1:1
    for iIdx = 1:n-1
      printf("%8.3f ",narray(jIdx,iIdx))
    end  
    printf("%8.3f\n",narray(jIdx,n))
  end

end
