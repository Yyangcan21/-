cmake_minimum_required(VERSION 3.19)
project(PuddleDetection LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets Sql LinguistTools)

qt_standard_project_setup()

# =============================================
# 可执行文件：水坑检测（训练 + 标注集成）
# =============================================
qt_add_executable(PuddleDetection WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    db_manager.cpp
    db_manager.h
    train.py
    auto_label.py
    quantize.py
    infer_quant.py
    annotationview.cpp
    annotationview.h
    inference_cpp.cpp
    inference_cpp.h
    losscurveview.cpp
    losscurveview.h
)

qt_add_translations(
    TARGETS PuddleDetection
    TS_FILES PuddleDetection_zh_CN.ts
)

target_include_directories(PuddleDetection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# onnxruntime_c_api.h 用了 MSVC 的 SAL 注解（_Frees_ptr_opt_ / _In_ / _Out_ 等），
# MinGW (g++) 不认这些宏。直接定义成空，避免 inference_cpp.cpp 编不过。
target_compile_definitions(PuddleDetection PRIVATE
    _Frees_ptr_opt_=
    _Frees_ptr_=
    _In_opt_=
    _Out_opt_=
    _Inout_opt_=
    _In_=
    _Out_=
    _Inout_=
    _In_z_=
    _Out_z_=
    _Inout_z_=
    _In_reads_\(x\)=
    _In_reads_bytes_\(x\)=
    _Out_writes_\(x\)=
    _Out_writes_bytes_\(x\)=
    _Outptr_=
    _Outptr_opt_=
    _Outptr_result_maybenull_=
    _Field_size_opt_=
    _Field_size_=
    _Check_return_=
    _Use_decl_annotations_=
    _ACRTIMP=
    _CRTIMP=
)

target_link_libraries(PuddleDetection
    PRIVATE
        Qt::Core
        Qt::Widgets
        Qt::Sql
        ${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.a
)

install(TARGETS PuddleDetection
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET PuddleDetection
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})

# ---- 自动复制 onnxruntime DLL 到 build 目录 ----
set(ORT_DLL_DIR "D:/Python/dog/.venv/Lib/site-packages/onnxruntime/capi")
if(EXISTS "${ORT_DLL_DIR}/onnxruntime.dll")
    add_custom_command(TARGET PuddleDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL_DIR}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:PuddleDetection>/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL_DIR}/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:PuddleDetection>/"
        COMMENT "Copying onnxruntime DLLs to build directory"
    )
else()
    message(WARNING "onnxruntime.dll not found at ${ORT_DLL_DIR}. C++ inference will not work.")
endif()

# ---- 自动复制 SQLite 驱动到 build 目录 ----
get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
set(_qt_plugins_dir "${_qt_bin_dir}/../plugins")

set(SQLITE_DRIVER "${_qt_plugins_dir}/sqldrivers/qsqlite.dll")
if(EXISTS ${SQLITE_DRIVER})
    add_custom_command(TARGET PuddleDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:PuddleDetection>/sqldrivers"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SQLITE_DRIVER}"
            "$<TARGET_FILE_DIR:PuddleDetection>/sqldrivers/"
        COMMENT "Copying qsqlite.dll to build directory"
    )
else()
    message(WARNING "qsqlite.dll not found at ${SQLITE_DRIVER}. Please copy it manually.")
endif()

file(GLOB_RECURSE DATASET_FILES "dataset/*")
