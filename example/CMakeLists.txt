cmake_minimum_required(VERSION 3.14...3.18)

project(cusz_example CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

## uncomment for pre-Turing
# set(CMAKE_CUDA_ARCHITECTURES 60 62 70 72)
## uncomment for V100
# set(CMAKE_CUDA_ARCHITECTURES 70)
## uncomment for Turing
set(CMAKE_CUDA_ARCHITECTURES 75)
## uncmoment for Ampere
# set(CMAKE_CUDA_ARCHITECTURES 80 86)
## uncmoment for A100
# set(CMAKE_CUDA_ARCHITECTURES 80)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR}/../src)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE SHARED)

add_executable(ex-api-huffcoarse src/ex_api_huffcoarse.cu)
target_link_libraries(ex-api-huffcoarse 
	PUBLIC
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	-lcuda
	-lcusparse)
set_target_properties(ex-api-huffcoarse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ex-api-csr src/ex_api_csr11.cu)
target_link_libraries(ex-api-csr 
	PUBLIC
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	-lcusparse)

add_executable(ex-api-pqlorenzo src/ex_api_pqlorenzo.cu)
target_link_libraries(ex-api-pqlorenzo 
	PUBLIC
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
 	-lcusparse)

add_executable(ex-defaultcompressor src/ex_api_defaultcompressor.cu)
target_link_libraries(ex-defaultcompressor  
	PUBLIC
	${CMAKE_SOURCE_DIR}/../Release/libcompress.a 
	${CMAKE_SOURCE_DIR}/../Release/libargp.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	-lcuda
	-lcusparse)
set_target_properties(ex-defaultcompressor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(spcompressor ${LIB_TYPE} ../src/sp_path.cu)
add_executable(ex-api-spline3 src/ex_api_spline3.cu)
target_link_libraries(ex-api-spline3 
	PUBLIC
	spcompressor 
	${CMAKE_SOURCE_DIR}/../Release/libhuff.a 
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	${CMAKE_SOURCE_DIR}/../Release/libsp.a 
	-lcuda
	-lcusparse)
set_target_properties(ex-api-spline3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(app-spline3-pred src/app_spline3_pred.cu)
target_link_libraries(app-spline3-pred
	${CMAKE_SOURCE_DIR}/../Release/libpq.a 
	-lcusparse)
# set_target_properties(app-spline3-pred PROPERTIES CUDA_SEPARABLE_COMPILATION ON)