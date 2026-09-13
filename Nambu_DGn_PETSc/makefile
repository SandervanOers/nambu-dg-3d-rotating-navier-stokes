
#include ${PETSC_DIR}/lib/petsc/conf/variables
#include ${PETSC_DIR}/lib/petsc/conf/rules
include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

initial_cond.o: initial_cond.cpp initial_cond.hpp
mesh_gen.o: mesh_gen.cpp mesh_gen.hpp
Legendre_Gauss_Lobatto.o: Legendre_Gauss_Lobatto.cpp Legendre_Gauss_Lobatto.hpp
HIGW.o: HIGW.cpp HIGW.hpp
Elements.o: Elements.cpp Elements.hpp
options.o: options.cpp options.hpp
test_cases.o: test_cases.cpp test_cases.hpp

Nambu02.o: Nambu02.cpp
OBJSNambu02 = Nambu02.o test_cases.o initial_cond.o mesh_gen.o Legendre_Gauss_Lobatto.o HIGW.o Elements.o options.o

Nambu02: ${OBJSNambu02}
		${CLINKER} -o Nambu02 ${OBJSNambu02} ${PETSC_LIB} ${PETSC_KSP_LIB} ${SLEPC_LIB}

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
clean::
	${RM} *.o *.d Nambu02

-include *.d
