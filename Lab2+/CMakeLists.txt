# CMakeList.txt: проект CMake для Lab2+; включите исходный код и определения,
# укажите здесь логику для конкретного проекта.
#

# Добавьте источник в исполняемый файл этого проекта.
add_executable (Lab2+ "Lab2+.cpp" "Lab2+.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Lab2+ PROPERTY CXX_STANDARD 20)
endif()

# TODO: Добавьте тесты и целевые объекты, если это необходимо.
find_package(OpenCV REQUIRED)

target_link_libraries(Lab2+ ${OpenCV_LIBS})