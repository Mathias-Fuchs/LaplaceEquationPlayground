cmake_minimum_required (VERSION 3.8)
find_package(Eigen3 CONFIG REQUIRED)

add_executable (abps2
"abps2.cpp"
)

target_link_libraries(abps2 PRIVATE
Eigen3::Eigen
)

# TODO: Add tests and install targets if needed.
