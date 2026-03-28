add_executable(ptx_web_runner
    ptx_web_runner.cpp
)

target_link_libraries(ptx_web_runner
    logger
    core
    decoder
    execution
    memory
    optimizer
    debugger
    host
    parser
    registers
)

target_include_directories(ptx_web_runner PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(ptx_web_runner PRIVATE /W4)
else()
    target_compile_options(ptx_web_runner PRIVATE -Wall -Wextra -pedantic)
endif()
