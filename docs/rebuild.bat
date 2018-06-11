REM rebuild docs shortcut
cd..
xcopy examples\*.png docs\figs /Y
cd docs
call make clean
call make html