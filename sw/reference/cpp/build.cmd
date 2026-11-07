@echo off
rem Build saracen_evm.dll (for the Python dashboard) and the original
rem magnification.exe, with MSVC 2022 + the official OpenCV 4.14 Windows SDK.
rem
rem OpenCV SDK location: %OPENCV_DIR% or %LOCALAPPDATA%\theTell\opencv\build
rem (extract opencv-4.14.0-windows.exe there:  opencv-4.14.0-windows.exe -o"%LOCALAPPDATA%\theTell" -y)

setlocal
set HERE=%~dp0
if "%OPENCV_DIR%"=="" set OPENCV_DIR=%LOCALAPPDATA%\theTell\opencv\build
set OCV_INC=%OPENCV_DIR%\include
set OCV_LIB=%OPENCV_DIR%\x64\vc16\lib\opencv_world4140.lib
set OCV_DLL=%OPENCV_DIR%\x64\vc16\bin\opencv_world4140.dll

if not exist "%OCV_INC%\opencv2\core.hpp" (echo [build] OpenCV headers not found in %OCV_INC% & exit /b 1)
if not exist "%OCV_LIB%" (echo [build] %OCV_LIB% not found & exit /b 1)

set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (echo [build] vcvars64.bat not found & exit /b 1)
call "%VCVARS%" >nul || exit /b 1

if not exist "%HERE%build" mkdir "%HERE%build"
pushd "%HERE%build"

echo [build] saracen_evm.dll
cl /nologo /O2 /EHsc /std:c++17 /MD /LD /I"%OCV_INC%" "%HERE%saracen_evm.cpp" "%OCV_LIB%" /Fe:saracen_evm.dll || (popd & exit /b 1)

echo [build] magnification.exe (original saracen main.cpp, unmodified)
cl /nologo /O2 /EHsc /std:c++17 /MD /I"%OCV_INC%" "%HERE%third_party\saracen\main.cpp" "%OCV_LIB%" /Fe:magnification.exe || (popd & exit /b 1)

if exist "%OCV_DLL%" (
  copy /y "%OCV_DLL%" . >nul
  echo [build] copied opencv_world4140.dll
) else (
  echo [build] WARNING: %OCV_DLL% missing - builds link, but cannot run until it is extracted
)
popd
echo [build] done: %HERE%build
endlocal
