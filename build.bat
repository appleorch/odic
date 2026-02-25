@echo off
echo Building DIC Tool .exe ...
pyinstaller dic_tool.spec
echo.
echo Build complete. Check the dist\ folder for dic_tool.exe
pause
