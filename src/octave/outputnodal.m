% Code to output a node based array for the oto problems
function outputnodal(nelx,nely,array,name)

  narray = reshape(array,nely+1,nelx+1);
  printf("\n%s :\n",name)
  for jIdx = nely+1:-1:1
    for iIdx = 1:nelx
      printf("%8.3f ",narray(jIdx,iIdx))
    end  
    printf("%8.3f\n",narray(jIdx,nelx+1))
  end

end
