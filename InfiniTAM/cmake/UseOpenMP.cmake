###################
# UseOpenMP.cmake #
###################

OPTION(WITH_OPENMP "Enable OpenMP support?" OFF)

IF(WITH_OPENMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
      message("hhg:OPENMP found!!!!!")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
  ADD_DEFINITIONS(-DWITH_OPENMP)
ENDIF()
