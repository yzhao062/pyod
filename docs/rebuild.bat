REM rebuild docs shortcut
REM only works for Windows
cd..
xcopy examples\*.png docs\figs /Y
xcopy notebooks\*.csv docs\tables\ /Y
cd docs
call make clean
call make html