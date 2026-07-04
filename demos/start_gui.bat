@echo off
chcp 65001 >nul 2>&1
title EEG GUI 启动器

REM ============================================================
REM EEG GUI 启动脚本
REM 用法: 双击运行 或 命令行 start_gui.bat
REM ============================================================

REM ---- 配置 ----
set GUI_DIR=%~dp0..\eeg_gui
set SRC_DIR=%GUI_DIR%\src
set LIB_DIR=%GUI_DIR%\lib
set CLASSPATH=.

REM 检查 lib 下是否有 jar
if exist "%LIB_DIR%\*.jar" (
    for %%f in ("%LIB_DIR%\*.jar") do set CLASSPATH=%CLASSPATH%;%%f
)

REM ---- 检查 JDK ----
where javac >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 javac，请确认 JDK 已安装并在 PATH 中
    echo 下载: https://adoptium.net/
    pause
    exit /b 1
)

REM ---- 检查源码 ----
if not exist "%SRC_DIR%\EEGMonitor.java" (
    echo [错误] 未找到 Java 源码: %SRC_DIR%\EEGMonitor.java
    pause
    exit /b 1
)

REM ---- 编译 ----
echo [1/2] 编译 Java 源码...
cd /d "%SRC_DIR%"
javac -encoding UTF-8 *.java 2>compile_err.txt
if %errorlevel% neq 0 (
    echo [错误] 编译失败:
    type compile_err.txt
    del compile_err.txt 2>nul
    pause
    exit /b 1
)
del compile_err.txt 2>nul
echo      编译成功

REM ---- 运行 ----
echo [2/2] 启动 GUI...
java -cp "%CLASSPATH%" EEGMonitor

if %errorlevel% neq 0 (
    echo.
    echo [错误] GUI 异常退出 (code %errorlevel%)
    pause
)
